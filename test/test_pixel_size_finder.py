from src.model.pixel_size_finder import PixelSizeFinder
from src.model.perspective_detector import PerspectiveDetector
from src.model.cropper import Cropper
import torchvision


def test_pixel_size_finder() -> None:
    scan = torchvision.io.read_image("test/test_data/data/ecg_data/10_1.png").float()
    mask = torchvision.io.read_image("test/test_data/data/ecg_data/10_1_mask.png").float()
    signal_probabilities = mask[2] / 255

    cropper = Cropper()
    alignment_params = PerspectiveDetector(num_thetas=300)(scan)
    source_points = cropper(signal_probabilities, alignment_params)
    aligned_image = cropper.apply_perspective(scan, source_points, fill_value=scan.mean().item())

    mm_per_pixel_x, mm_per_pixel_y = PixelSizeFinder(plot=True)(aligned_image)

    assert abs(mm_per_pixel_x - 1.1241) < 0.001
    assert abs(mm_per_pixel_y - 1.1331) < 0.001
