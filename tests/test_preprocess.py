"""Input images are cover-resized and center-cropped, never stretched (SPEC §11)."""

from PIL import Image, ImageDraw

from src.backend import preprocess_image

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def close(pixel, color, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(pixel, color, strict=True))


def blue_bbox(image):
    mask = image.point(lambda v: 255 if v > 128 else 0).split()[2]
    return mask.getbbox()


def test_landscape_example_from_spec():
    # 1200x800 -> 1024x1024: resize to 1536x1024, crop 256px from each side.
    src = Image.new("RGB", (1200, 800), GREEN)
    draw = ImageDraw.Draw(src)
    draw.rectangle((0, 0, 149, 799), fill=RED)  # left band, cropped away
    draw.rectangle((1050, 0, 1199, 799), fill=RED)  # right band, cropped away
    draw.rectangle((500, 300, 699, 499), fill=BLUE)  # centered 200x200 square

    out = preprocess_image(src, 1024, 1024)

    assert out.size == (1024, 1024)
    assert out.mode == "RGB"
    for xy in [(2, 2), (1021, 2), (2, 1021), (1021, 1021)]:
        assert close(out.getpixel(xy), GREEN), xy
    left, top, right, bottom = blue_bbox(out)
    # Square stays square (~256px after x1.28 scale), i.e. no distortion.
    assert abs((right - left) - (bottom - top)) <= 2
    assert abs((right - left) - 256) <= 3


def test_portrait_source_crops_top_and_bottom():
    src = Image.new("RGB", (400, 800), GREEN)
    draw = ImageDraw.Draw(src)
    draw.rectangle((0, 0, 399, 99), fill=RED)
    draw.rectangle((0, 700, 399, 799), fill=RED)
    draw.rectangle((150, 350, 249, 449), fill=BLUE)

    out = preprocess_image(src, 512, 768)

    # scale = max(512/400, 768/800) = 1.28 -> 512x1024, crop 128px top and bottom
    assert out.size == (512, 768)
    assert close(out.getpixel((256, 2)), GREEN)
    assert close(out.getpixel((256, 765)), GREEN)
    left, top, right, bottom = blue_bbox(out)
    assert abs((right - left) - (bottom - top)) <= 2


def test_small_image_is_upscaled_to_cover():
    out = preprocess_image(Image.new("RGB", (100, 50), GREEN), 512, 512)
    assert out.size == (512, 512)


def test_mode_conversion_rgba_and_grayscale():
    assert preprocess_image(Image.new("RGBA", (300, 300)), 256, 256).mode == "RGB"
    assert preprocess_image(Image.new("L", (300, 300)), 256, 256).mode == "RGB"


def test_exif_orientation_is_applied(tmp_path):
    # Stored 400x200 landscape: left half red, right half blue.
    stored = Image.new("RGB", (400, 200), RED)
    ImageDraw.Draw(stored).rectangle((200, 0, 399, 199), fill=BLUE)
    exif = stored.getexif()
    exif[0x0112] = 6  # display rotated 90 degrees clockwise
    path = tmp_path / "rotated.jpg"
    stored.save(path, exif=exif, quality=95)

    # Displayed as 200x400 portrait: red on top, blue on bottom.
    # Covering 256x256 -> 256x512, crop rows 128..384: top half red, bottom half blue.
    out = preprocess_image(path, 256, 256)
    assert out.size == (256, 256)
    assert close(out.getpixel((128, 20)), RED)
    assert close(out.getpixel((128, 235)), BLUE)


def test_exact_size_is_untouched():
    src = Image.new("RGB", (512, 512), GREEN)
    out = preprocess_image(src, 512, 512)
    assert out.size == (512, 512)
    assert close(out.getpixel((0, 0)), GREEN)
