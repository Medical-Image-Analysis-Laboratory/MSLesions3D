from monai.inferers.utils import _get_scan_interval
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from monai.data.utils import dense_patch_slices
from monai.utils import (
    PytorchPadMode,
    convert_data_type,
    fall_back_tuple,
    look_up_option,
    optional_import,
)
from utils import find_jaccard_overlap3d

tqdm, _ = optional_import("tqdm", name="tqdm")


def _convert_coordinates(coords, roi_shape, image_shape, patch_position, num_spatial_dims):
    """
    Converts patch coordinates to image coordinates.

    Args:
        coords: list (size <batch_size>) of torch.Tensor coordinates for every image in the batch
        roi_shape: shape of the roi
        image_shape: shape of the image
        patch_position: position of the patch (list of slices)
        num_spatial_dims: number of spatial dimensions

    Returns:
        list (size <batch_size>) of torch.Tensor converted coordinates for every image in the batch
    """
    device = coords[0].device
    actual_coords_in_patch = [c * torch.Tensor(roi_shape * 2).to(device) for c in coords]
    patch_top_left_corner = [
        torch.Tensor(tuple([p[-num_spatial_dims:][i].start for i in range(num_spatial_dims)]) * 2).to(device)
        for p in patch_position
    ]

    actual_coords_in_image = [
        (a + p) / torch.Tensor(image_shape * 2).to(device)
        for a, p in zip(actual_coords_in_patch, patch_top_left_corner)
    ]
    return actual_coords_in_image


def sliding_window_inference_object_detection(
        inputs: torch.Tensor,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
        overlap: float = 0.25,
        padding_mode: str = "constant",
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        progress: bool = False,
        score_thresh: float = 0.5,
        max_overlap: float = 0.5,
        image_top_k: int = 100,
        *args: Any,
        **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.
    Adapted from `monai.inferers.utils.sliding_window_inference` to support (only) object detection.

    The outputs of `predictor` should be (bboxes, scores, labels) <(torch.Tensor, torch.Tensor, torch.Tensor)>.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default, the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        score_thresh: threshold for object detection score.
        max_overlap: maximum overlap for non-maximum suppression.
        image_top_k: maximum number of detections to keep after non-maximum suppression.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=padding_mode, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Perform predictions
    output_bboxes_list = {idx: torch.empty((0, 6)).to(device) for idx in range(batch_size)}
    output_scores_list = {idx: torch.empty(0).to(device) for idx in range(batch_size)}
    output_labels_list = {idx: torch.empty(0).to(device) for idx in range(batch_size)}

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]  # ex: [slice(0, 1, None), slice(None, None, None), # slice(0, 64, None), # slice(0, 64, None)]

        img_ids = [idx // num_win for idx in slice_range]  # img ids for each patch

        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]
        ).to(sw_device)
        coords, labels, scores = predictor(window_data, *args, **kwargs)  # batched patch outputs

        # Convert patch coordinates to the image coordinates
        with torch.no_grad():
            image_coords = _convert_coordinates(coords, roi_size, image_size, unravel_slice, num_spatial_dims)

        # Add predicted bounding boxes to the lists
        for i, img_id in enumerate(img_ids):
            output_bboxes_list[img_id] = torch.cat([output_bboxes_list[img_id], image_coords[i]])
            output_scores_list[img_id] = torch.cat([output_scores_list[img_id], scores[i]])
            output_labels_list[img_id] = torch.cat([output_labels_list[img_id], labels[i]])

    # Non-Maximum Suppression
    for i in range(batch_size):
        output_bboxes_list[i], output_scores_list[i], output_labels_list[i] = nms(
            output_bboxes_list[i],
            output_scores_list[i],
            output_labels_list[i],
            score_thresh,
            max_overlap,
            image_top_k,
        )

    # Stack the results into a single tensor
    output_bboxes = torch.stack([output_bboxes_list[i] for i in range(batch_size)])
    output_scores = torch.stack([output_scores_list[i] for i in range(batch_size)])
    output_labels = torch.stack([output_labels_list[i] for i in range(batch_size)])

    return output_bboxes, output_scores, output_labels


def nms(bboxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        min_score: float,
        max_overlap: float,
        top_k: int):
    """
    Performs non-maximum suppression (NMS) on the bounding boxes according to their intersection-over-union (IoU)
    and their scores.

    Parameters:
        bboxes (torch.Tensor): A tensor of shape (n, 6) containing the bounding boxes coordinates.
        scores (torch.Tensor): A tensor of shape (n,) containing the scores of the bounding boxes.
        labels (torch.Tensor): A tensor of shape (n,) containing the labels of the bounding boxes.
        min_score (float): Minimum score for a bounding box to be considered for NMS.
        max_overlap (float): Maximum overlap between two bounding boxes to be considered as predicting the same object.
        top_k (int): Number of top scoring bounding boxes to keep for each image.

    Returns:
        image_boxes (torch.Tensor): A tensor of shape (m, 6) containing the filtered bounding boxes coordinates.
        image_labels (torch.Tensor): A tensor of shape (m,) containing the labels of the filtered bounding boxes.
        image_scores (torch.Tensor): A tensor of shape (m,) containing the scores of the filtered bounding boxes.
    """
    device = bboxes.device

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(1, int(max(torch.unique(labels)).cpu()) + 1):
        # Keep only predicted boxes and scores where scores for this class are above the minimum score
        class_scores = scores[labels == c]  # (xxx)

        score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
        n_above_min_score = score_above_min_score.sum().item()
        if n_above_min_score == 0:
            continue
        class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= xxx
        class_bboxes = bboxes[score_above_min_score]  # (n_qualified, 6)

        # Sort predicted boxes and scores by scores
        class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
        class_bboxes = class_bboxes[sort_ind]  # (n_min_score, 6)

        n_above_min_score = min(10 * top_k, n_above_min_score)
        class_scores = class_scores[:n_above_min_score]
        class_bboxes = class_bboxes[:n_above_min_score]

        # Find the overlap between predicted boxes
        overlap = find_jaccard_overlap3d(class_bboxes, class_bboxes)  # (n_qualified, n_min_score)

        # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
        # 1 implies suppress, 0 implies don't suppress
        suppress = torch.zeros(n_above_min_score).bool().to(device)

        # Consider each box in order of decreasing scores
        for box in range(class_bboxes.size(0)):
            # If this box is already marked for suppression
            if suppress[box] == 1:
                continue

            # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
            # Find such boxes and update suppress indices
            suppress = suppress | (overlap[box] > max_overlap)

            # Don't suppress this box, even though it has an overlap of 1 with itself
            suppress[box] = 0

        # Store only unsuppressed boxes for this class
        image_boxes.append(class_bboxes[~suppress])
        image_labels.append(
            torch.LongTensor(
                (~suppress).sum().item() * [c]).to(device)
        )
        image_scores.append(class_scores[~suppress])

    # If no object in any class is found, store a placeholder for 'background'
    if len(image_boxes) == 0:
        image_boxes.append(torch.FloatTensor([[0., 0., 0., 1., 1., 1.]]).to(device))
        image_labels.append(torch.LongTensor([0]).to(device))
        image_scores.append(torch.FloatTensor([0.]).to(device))

    # Concatenate into single tensors
    image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 6)
    image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    n_objects = image_scores.size(0)

    # Keep only the top k objects
    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]  # (top_k)
        image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 6)
        image_labels = image_labels[sort_ind][:top_k]  # (top_k)

    return image_boxes, image_scores, image_labels


class ObjectDetector:
    def __init__(self, model_path, model_class, min_score=0.5, max_overlap=0.5, top_k=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class.load_from_checkpoint(model_path, min_score=min_score).to(self.device)
        self.model.eval()
        self.min_score = min_score
        self.max_overlap = max_overlap
        self.top_k = top_k

    def __call__(self, image):
        if type(image) == np.ndarray:
            image = np.expand_dims(image, axis=0)  # add channel dimension
            image = np.expand_dims(image, axis=0)  # add batch dimension
            image = torch.from_numpy(image)

        # Move to default device
        image = image.to(self.device)

        with torch.no_grad():
            # Forward prop.
            predicted_locs, predicted_scores = self.model(image)

            # Detect objects in model output (+ perform NMS)
            det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs,
                                                                          predicted_scores,
                                                                          min_score=self.min_score,
                                                                          max_overlap=self.max_overlap,
                                                                          top_k=self.top_k)

        return det_boxes, det_labels, det_scores


if __name__ == '__main__':
    import nibabel as nib

    image = nib.load(f"/home/wynen/data/test/multiple_objects/one_class/images/sub-0000_image.nii.gz").get_fdata()
    model_path = r'/home/wynen/data/model_test.ckpt'
    from ssd3d import LSSD3D
    od = ObjectDetector(model_path, model_class=LSSD3D)

    print()
