# Pytorch AI Dungeon2

A Fork of Nick Walton's [AI Dungeon2](https://github.com/AIDungeon/AIDungeon) and pytorch as a backend. 

Uses prompts and play.py from the [Clover edition](https://github.com/cloveranon/Clover-Edition)

I did this because tensorflow is annoying to compile for my xeon processor an it was actually faster to port the generation code.

No colab yet.

If you want the converted model, just ask in the issues and I'll make it available.

## Changes

- use half precision for smaller model
- use pytorch
- added suggested actions (a bit messy and poor right now)
- 50% of the time an action is attempted ("You X" vs "You try to X") to make it harder

