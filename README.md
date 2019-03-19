# Quality-Estimation2
机器翻译子任务-翻译质量评价-在BERT模型后面加上Bi-LSTM进行fine-tuning<br>

## 简介
翻译质量评价（Quality Estimation,QE）是机器翻译领域中的一个子任务，大致可分为 Sentence-level QE，Word-level QE，Phrase-level QE，详情可参考WMT(workshop machine translation)比赛官网 http://www.statmt.org/wmt17/quality-estimation-task.html 。本项目针对 Sentence-level QE，在BERT模型后面加上Bi-LSTM进行fine-tuning，代码参考了 https://github.com/huggingface/pytorch-pretrained-BERT 。 由于 wmt18-qe 的测试集标签没有公布，本项目仅在 wmt17-qe 数据集上进行实验。

## 实验需要的包
PyTorch 0.4.1/1.0.0;<br>
python3;

## 实验步骤
1、准备数据，下载17年wmt sentence level的数据，将数据放置在 ./examples/QE 文件夹下，数据文件示例见QE文件夹;<br>
2、下载bert预训练模型，放到 ./pretrain-models 文件夹并解压，这里用到的预训练模型是：BERT-Base, Multilingual Cased (New, recommended): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters，<br>
可以到这里下载：https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz ;<br>
3、运行run_qe.sh进行fine-tuning;<br>

## 实验结果
本人使用pytorch版本的bert进行fine-tuning，但是实验结果很不好，en_de_pearson只能到0.3，训练过程的loss曲线一直震荡(图1)，**到目前为止还没有找到原因（求大佬解答）**。下面表格中的实验结果是用朋友的代码（基于tensorflow版本的bert）跑出的结果，供大家参考。

|Data|Pearson’s|MAE|RMSE|Spearman’s|
|:---|:---|:---|:---|:---|
|test 2017 en-de|0.6791|0.1036|0.1517|0.7103|
|state of the art(Single)|0.6837|0.1001|0.1441|0.7091|
|test 2017 de-en| 0.7239|0.0834|0.1392|0.6869|
|state of the art(Single)|0.7099|0.0927|0.1394|0.6424|

注：state of the art 参考论文：[“Bilingual Expert” Can Find Translation Errors](https://arxiv.org/pdf/1807.09433.pdf) ;<br>
<br>
<img src="https://github.com/xlniu/Quality-Estimation2/blob/master/pretrain-models/loss.png" width="450" height="288" /> <br>
** **  图1 基于pytorch-bert进行fine-tuning的loss输出
