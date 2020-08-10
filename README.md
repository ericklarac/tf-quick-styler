# TF Quick Styler

Based on the model code in [magenta](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) and the publication:

[Exploring the structure of a real-time, arbitrary neural artistic stylization
network](https://arxiv.org/abs/1705.06830).
_Golnaz Ghiasi, Honglak Lee,
Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens_,
Proceedings of the British Machine Vision Conference (BMVC), 2017.

Base code taken from TF-Hub: Fast Style Transfer for Arbitrary Styles.ipynb

_Original file is located at [the this notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb)_

### List of predefined style:

- kanagawa_great_wave
- kandinsky_composition_7
- hubble_pillars_of_creation
- van_gogh_starry_night
- turner_nantes
- munch_scream
- picasso_demoiselles_avignon
- picasso_violin
- picasso_bottle_of_rum
- fire
- derkovits_woman_head
- amadeo_style_life
- derkovtis_talig
- amadeo_cardoso

### Commands

#### Using a predifined style

```
$ python quick-styler.py --content-image-url https://image-cdn.hypb.st/https%3A%2F%2Fhypebeast.com%2Fimage%2F2018%2F09%2Fodell-beckham-jr-facebook-documentary-series-i-am-more-obj-0.jpg --output-file-name styled-image-cli --style-name turner_nantes
```

![Styled image with the command above](./images/styled-image-cli.png)

#### Using a URL style

```
$ python quick-styler.py --content-image-url https://image-cdn.hypb.st/https%3A%2F%2Fhypebeast.com%2Fimage%2F2018%2F09%2Fodell-beckham-jr-facebook-documentary-series-i-am-more-obj-0.jpg --output-file-name styled-image-cli-url --style-url https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Great_Wave_off_Kanagawa2.jpg/800px-Great_Wave_off_Kanagawa2.jpg
```

![Styled image with the command above](./images/styled-image-cli-url.png)

### Help

Run the following command to get the help of the styler:

```
$ python quick-styler.py --help
```

#### Copyright 2019 The TensorFlow Hub Authors.

#### Licensed under the Apache License, Version 2.0 (the "License");
