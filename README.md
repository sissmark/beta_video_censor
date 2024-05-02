# beta video censor

New tool to censor your videos! https://github.com/sissmark/beta_video_censor…Looking for input and testers! #sissysafe #betasafe #sissycensored #betacensored #loserporn #censoredporn

This script will censor female nsfw parts in videos using the nudenet ai model.
I tested it using Python 3.12.

Once downloaded

install requirements: #pip install -r requirements.txt

run the script: #python censor_video_gui.py

This should open a window allowing you to select a video file.
Once selected the script will show a preview of the censoring process and when finished store the video as [video_name]_censored.mp4

Things on my mind:

- Add support for audio, currently no audio is copied to the censored video;
- Add options to customize censorship, i.e what should and shouldn't be censored and in which way;
- More censorship options, i.e. blur, pixelation, mosaic, image overlay, etc.
- Custom audio filter options, detect NSFW words and remove them, convert audio to lower quality or 'muffled' (Like listening from another room)
- The usage of a AI image model to remove or add certain parts
- A better gui / install / run experience.

Let me know what you think!
