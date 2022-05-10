# Learning When to Translate for Streaming Speech
This is a PyTorch implementation for the ACL 2022 main conference paper [ Learning When to Translate for Streaming Speech ](https://arxiv.org/abs/2109.07368).

## Data Processing
Take German for example.
Firstly, download [MuST-C v1.0](https://ict.fbk.eu/must-c/) archive `MUSTC_v1.0_en-de.tar.gz` to the `${MUSTC_ROOT}` path, and uncompress it:
```shell script
LANG=de
MUSTC_ROOT=/path/data/en-${LANG}$
tar -xzvf MUSTC_v1.0_en-de.tar.gz
```
Then, run the script to prepare data manifest.
```shell script
python3 examples/speech_to_text/prep_mustc_data_raw.py --data-root ${MUSTC_ROOT} \
  --tgt-lang ${LANG}
```

The generated `.tsv` should be expanded with the field of source language text and doubled with asr task. Here's some examples from the `.tsv` file.

```
id      audio   n_frames        tgt_text        speaker tgt_lang        src_text        src_lang
ted_2529_66     /xxx/en-de/data/train/wav/ted_2529.wav:9517120:61760      61760   Ich hatte den Vorteil einer Perspektive von dieser Breite.  spk.2529        de      I had the benefit of a spectrum this wide.      en
ted_1257_134    /xxx/en-de/data/train/wav/ted_1257.wav:13876160:80960     80960   And outside the library, I wanted to make a place to cultivate your mind.   spk.1257        en      And outside the library, I wanted to make a place to cultivate your mind.       en
ted_362_30      /xxx/en-de/data/train/wav/ted_362.wav:488959:156960       156960  Ich lebe genau hier im West Village, die Rauchwolke wurde zum Glück westwärts geweht, weg von uns.  spk.362 de      I live right there in the West Village, so the plume was luckily blowing west, away from us.        en
...
ted_526_7       /xxx/en-de/data/train/wav/ted_526.wav:16538720:19360      19360   It can also happen in the brain.    spk.526 en      It can also happen in the brain.        en
ted_190_62      /xxx/en-de/data/train/wav/ted_190.wav:7045920:47360       47360   Simple question: if you can't read and write, how do you manage your contact information?   spk.190 en      Simple question: if you can't read and write, how do you manage your contact information?   en
ted_1771_81     /xxx/en-de/data/train/wav/ted_1771.wav:9624320:25600      25600   This is my message to you. spk.1771 en      This is my message to you.      en
```

The preprocessed directory `${MUSTC_ROOT}` should look like as follows:

```
.
├── en-de
│   ├── config_wave.yaml
│   ├── data
│   ├── dev_wavecif_joint.tsv
│   ├── docs
│   ├── segment
│   ├── spm_unigram10000_st.model
│   ├── spm_unigram10000_st.txt
│   ├── spm_unigram10000_st.vocab
│   ├── train_wavecif_joint.tsv
│   ├── tst-COMMON_wavecif_joint.tsv
│   ├── tst-HE_wavecif_joint.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

The generated `config_wave.yaml` should look like as follows:

```
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: spm_unigram10000_st.model
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
vocab_filename: spm_unigram10000_st.txt
use_audio_input: true
prepend_tgt_lang_tag: true
```

## Training

+ Training with multitask learning.

```shell script
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_wave.yaml \
  --train-subset train_wavecif_joint \
  --valid-subset dev_wavecif_joint \
  --save-dir /path/${LANG}/pretrain \
  --max-tokens 3200000  \
  --update-freq 1 \
  --max-update 3200000 \
  --task speech_to_text_wav2vec_cif \
  --criterion qua_ce_acc_v2 \
  --arch convtransformer_espnet_wav2vec_cif \
  --w2v2-model-path /path/wav2vec_small.pt \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 25000 \
  --clip-norm 10.0 \
  --seed 1 \
  --ddp-backend=no_c10d \
  --keep-best-checkpoints 10 \
  --best-checkpoint-metric accuracy \
  --maximize-best-checkpoint-metric \
  --patience 15 \
  --max-source-positions 3200000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.0 --activation-dropout 0.1 --attention-dropout 0.1 \
  --encoder-layers 8 \
  --ignore-prefix-size 1 --log-interval 20  --fp16 \
  --load-pretrained-encoder-from ${pretrain_ckpt} --load-pretrained-decoder-from ${pretrain_ckpt} 
```

We use the pre-trained [ Wav2vec 2.0 ](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) as the acoustic encoder.

+ Fine-tuning with monotonic segmentation module.

```shell script
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_wave.yaml \
  --train-subset train_wavecif_joint \
  --valid-subset dev_wavecif_joint \
  --save-dir /path/${LANG}/finetune/ \
  --max-tokens 3200000  \
  --update-freq 1 \
  --max-update 3200000 \
  --task speech_to_text_wav2vec_cif \
  --criterion qua_ce_acc_v2 \
  --arch convtransformer_espnet_wav2vec_cif \
  --w2v2-model-path /path/wav2vec_small.pt \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --clip-norm 10.0 \
  --seed 1 \
  --ddp-backend=no_c10d \
  --keep-best-checkpoints 10 \
  --best-checkpoint-metric accuracy \
  --maximize-best-checkpoint-metric \
  --patience 15 \
  --max-source-positions 3200000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.0 --activation-dropout 0.1 --attention-dropout 0.1 \
  --encoder-layers 8 \
  --ignore-prefix-size 1 --log-interval 20  --fp16 \
  --load-pretrained-encoder-from /path/${LANG}/pretrain/checkpoint.pt \
  --load-pretrained-decoder-from /path/${LANG}/pretrain/checkpoint.pt
```



## Evaluation
### Offline Translation
Our released models ([En-De](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/mostt/en-de.pt) and [En-Fr](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/mostt/en-fr.pt)) can be downloaded to test the evaluation directly.

```shell script
fairseq-generate ${MUSTC_ROOT} \
  --config-yaml config_wave.yaml \
  --gen-subset tst-COMMON_wavecif_joint_st \
  --task speech_to_text_wav2vec_cif \
  --path /path/${LANG}/finetune/checkpoint.pt \
  --max-tokens 3200000 \
  --beam 5 \
  --scoring sacrebleu \
  --max-source-positions 3200000 \
  --prefix-size 1
```

### Streaming Translation

Note that the offline models need to be converted to support streaming translation task. Our model ([En-De](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2022/mostt/en-de_online.pt) can be downloaded to test streaming translation.
+ Prefix-decision 
```shell script
lagging=5
fixed-pre-decision-ratio=7
simuleval --agent fairseq/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent_wav2vec.py \
  --source /path/data/tst-COMMON.wavurl \
  --target /path/data/tst-COMMON.${LANG} \
  --data-bin /path/data/en-${LANG}/ \
  --config config_wave.yaml \
  --model-path /path/${LANG}/finetune/checkpoint.pt \
  --output /path/${LANG}/finetune/simuleval/ \
  --waitk-lagging ${lagging} \
  --fixed-pre-decision-ratio ${fixed-pre-decision-ratio} \
  --scores \
  --port 1234 \
  --prefix-size 1 \
  --lang de
```

+ Dynamic-decision

```shell script
simuleval --agent fairseq/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent_wav2vec_cif.py \
  --source /path/data/tst-COMMON.wavurl \
  --target /path/data/tst-COMMON.${LANG} \
  --data-bin /path/data/en-${LANG}/ \
  --config config_wave.yaml \
  --model-path /path/${LANG}/finetune/checkpoint.pt \
  --output /path/${LANG}/finetune/simuleval/ \
  --scores \
  --max-source-positions 3200000 \
  --port 1234 \
  --prefix-size 1 \
  --lang de
```

## Citation
Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.

```
@inproceedings{dong-etal-2022-Learning,
	title = {Learning When to Translate for Streaming Speech},
	author = {Qianqian Dong, Yaoming Zhu, Mingxuan Wang, Lei Li},
	booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	year = {2022},
}
```
