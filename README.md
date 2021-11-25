# METER

Training/evaluation commands will come soon.

### Pre-trained Checkpoints

Here are the pre-trained models of METER-CLIP16-RoBERTa:
1. METER-CLIP16-RoBERTa pre-trained on GCC+SBU+COCO+VG [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_pretrain.ckpt)
2. METER-CLIP16-RoBERTa fine-tuned on VQAv2 [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_vqa.ckpt)
3. METER-CLIP16-RoBERTa fine-tuned on NLVR2 [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_nlvr2.ckpt)
4. METER-CLIP16-RoBERTa fine-tuned on SNLI-VE [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_snli.ckpt)
5. METER-CLIP16-RoBERTa fine-tuned on Flickr30k IR/TR [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_flickr.ckpt)
6. METER-CLIP16-RoBERTa fine-tuned on COCO IR/TR [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_coco.ckpt)


### Citation

```
@article{dou2021meter,
  title={An Empirical Study of Training End-to-End Vision-and-Language Transformers},
  author={Dou, Zi-Yi and Xu, Yichong and Gan, Zhe and Wang, Jianfeng and Wang, Shuohang and Wang, Lijuan and Zhu, Chenguang and Zhang, Pengchuan and Yuan, Lu and Peng, Nanyun and Liu, Zicheng and Zeng, Michael},
  journal={arXiv},
  year={2021},
  url={https://arxiv.org/abs/2111.02387},
}
```

### Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).
