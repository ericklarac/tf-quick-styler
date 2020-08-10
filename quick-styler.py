# Setup
import os
import typer
import logging
import PIL.Image
import functools
import coloredlogs

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from helper import crop_center, load_image_path, load_image_url, tensor_to_image, style_urls, is_url

logger = logging.getLogger(__name__)
logger.name = "Quick Styler"
coloredlogs.install(
    level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s')

"""
# print("TF Version: ", tf.__version__)
# print("TF-Hub version: ", hub.__version__)
# print("Eager mode enabled: ", tf.executing_eagerly())
# print("GPU available: ", tf.test.is_gpu_available())
"""

app = typer.Typer()


@app.command()
def style_images(content_image_url='', style_url='', output_file_name="styled_image", style_name="picasso_violin"):
    """Style images using Tensor Flow Fast Style Transfer for Arbitrary Styles

    Parameters

    ----------

    content_image_url : str
        Provide a full path to the image from your computer, or an URL starting with http

    style_url : str
        Provide a full path to the style image from your computer, or an URL starting with http

    output_file_name : str
        Name for the output file, by default styled_image

    style_name : str
        Style to add to the image, choice one:\n
          * 'kanagawa_great_wave'\n
          * 'kandinsky_composition_7'\n
          * 'hubble_pillars_of_creation'\n
          * 'van_gogh_starry_night'\n
          * 'turner_nantes'\n
          * 'munch_scream'\n
          * 'picasso_demoiselles_avignon'\n
          * 'picasso_violin'\n
          * 'picasso_bottle_of_rum'\n
          * 'fire'\n
          * 'derkovits_woman_head'\n
          * 'amadeo_style_life'\n
          * 'derkovtis_talig'\n
          * 'amadeo_cardoso'\n
        By default "picasso_violin"
    """
    if not content_image_url or not (content_image_url.endswith("png") or content_image_url.endswith("jpg")):
        print(f"Content image path not properly formated: {content_image_url}")
        return
    elif not content_image_url and not(content_image_url.startswith("http") or content_image_url[2] == ":"):
        print(f"Content image URL not properly formated: {content_image_url}")
        return

    content_image_url = content_image_url
    # "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Great_Wave_off_Kanagawa2.jpg/800px-Great_Wave_off_Kanagawa2.jpg"

    style_image_url = ''
    if style_url:
        if not ((style_url.endswith("png") or style_url.endswith("jpg")) and style_url.startswith("http")):
            print(f"Style URL not properly formated: {style_url}")
            return
        style_image_url = style_url
    else:
        if style_name not in style_urls:
            print(f"{style_name} is not a predefined style")
            return

        logger.info(
            f"Style was not provided, using default style ({style_name})")

    # The content image size can be arbitrary.
    output_image_size = 512  # @param {type:"integer"}
    content_img_size = (output_image_size, output_image_size)
    content_image = load_image_url(content_image_url, content_img_size) if is_url(
        content_image_url) else load_image_path(content_image_url, content_img_size)

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    style_image_size = 256

    logger.debug(f"Adding style to the image")

    if style_image_url:
        stylized_image = load_image_url(style_image_url) if is_url(
            style_image_url) else load_image_path(style_image_url)
        stylized_image = tf.nn.avg_pool(
            stylized_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
        stylized_image = hub_module(tf.convert_to_tensor(
            content_image), tf.convert_to_tensor(stylized_image))[0]
    else:
        style_images = {k: load_image_url(
            v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
        style_images = {k: tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[
                                          1, 1], padding='SAME') for k, style_image in style_images.items()}

        style_name = style_name
        stylized_image = hub_module(tf.convert_to_tensor(
            content_image), tf.convert_to_tensor(style_images[style_name]))[0]

    logger.debug("Applying last details")

    file_name = output_file_name + ".png"
    tensor = tf.Variable(stylized_image, validate_shape=False)
    tensor_to_image(tensor).rotate(270).save(file_name)
    logger.debug(f"Image saved successfully at {file_name}.")


if __name__ == "__main__":
    app()
