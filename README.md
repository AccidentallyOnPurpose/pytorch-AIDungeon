# Pytorch AI Dungeon2

A Fork of Nick Walton's [AI Dungeon2](https://github.com/AIDungeon/AIDungeon) and pytorch as a backend. 

Uses prompts and play.py from the [Clover edition](https://github.com/cloveranon/Clover-Edition)

I did this because tensorflow is annoying to compile for my xeon processor an it was actually faster to port the generation code.

No colab yet.

If you want the converted model, just ask in the issues and I'll make it available.

Content warning: This model is trained on the internet which means there will be lots of toxic and offensive content along with the funny and wierd. 

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

# Model

You can convert the model using `scripts/convert_gpt2_model.py`

```
export OPENAI_GPT2_CHECKPOINT_PATH=../generator/gpt2/models/model_v5
export PYTORCH_DUMP_OUTPUT=../generator/gpt2/models/model_v5_pytorch
python convert_gpt2_model.py \
    --gpt2_checkpoint_path $OPENAI_GPT2_CHECKPOINT_PATH \
    --pytorch_dump_folder_path $PYTORCH_DUMP_OUTPUT \
    --gpt2_config_file ./aidungeonv2_model_v5_config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt -o $PYTORCH_DUMP_OUTPUT/merges.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json -o $PYTORCH_DUMP_OUTPUT/vocab.json
```


## magnet

[torrent file](https://github.com/AccidentallyOnPurpose/pytorch-AIDungeon/blob/f692e39d84/generator/gpt2/models/model_v5_pytorch.torrent?raw=true)

And here is a magnet:

<a href="magnet:?xt=urn:btih:17dcfe3d12849db04a3f64070489e6ff5fc6f63f&dn=model_v5_pytorch&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce&tr=udp%3a%2f%2fopen.stealth.si%3a80%2fannounce&tr=udp%3a%2f%2fp4p.arenabg.com%3a1337%2fannounce&tr=udp%3a%2f%2ftracker.coppersurfer.tk%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.cyberia.is%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.moeking.me%3a6969%2fannounce&tr=udp%3a%2f%2f9.rarbg.me%3a2710%2fannounce&tr=udp%3a%2f%2ftracker3.itzmx.com%3a6961%2fannounce">magnet link to pytorch model torrent</a>

    ```magnet:?xt=urn:btih:17dcfe3d12849db04a3f64070489e6ff5fc6f63f&dn=model_v5_pytorch&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce&tr=udp%3a%2f%2fopen.stealth.si%3a80%2fannounce&tr=udp%3a%2f%2fp4p.arenabg.com%3a1337%2fannounce&tr=udp%3a%2f%2ftracker.coppersurfer.tk%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.cyberia.is%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.moeking.me%3a6969%2fannounce&tr=udp%3a%2f%2f9.rarbg.me%3a2710%2fannounce&tr=udp%3a%2f%2ftracker3.itzmx.com%3a6961%2fannounce```


## s3

Here is a temporary s3 download, I will disable if it costs to much

https://g5gh6hdfg76ikyfghdfgerd.s3-us-west-2.amazonaws.com/model_v5_pytorch.tar.gz

```
> md5sum model_v5_pytorch.tar.gz
5ec9502a27b087d13e9d9e9bfcff02b2  model_v5_pytorch.tar.gz
```
