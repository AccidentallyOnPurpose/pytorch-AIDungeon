# Pytorch AI Dungeon2

A Fork of Nick Walton's [AI Dungeon2](https://github.com/AIDungeon/AIDungeon) and pytorch as a backend. 

Uses prompts and play.py from the [Clover edition](https://github.com/cloveranon/Clover-Edition)

I did this because tensorflow is annoying to compile for my xeon processor an it was actually faster to port the generation code.

No colab yet.

If you want the converted model, just ask in the issues and I'll make it available.

## Screenshot

![](http://i.imgur.com/4Ox8zDX.png)

## Changes

- user:
  - added suggested actions from the AI player
  - roll a d20 for speech or action to make it harder
    - d01: You fail to X
    - dX: You try to X
    - d20: You successfully X
  - use Clover edition ui, prompts, config
- technical:
  - use half precision for smaller model, (but this might lead to lower quality, I need to test more)
  - better logging
  - use pytorch
  - set top_k to zero and just use top p
  - change clover config file to yaml

