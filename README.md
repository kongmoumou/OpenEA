# [A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/OpenEA/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/OpenEA/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-orange.svg?style=flat-square)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/VLDB%202020-PDF-yellow.svg?style=flat-square)](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)

> 实体对齐寻求在不同的知识图（KG）中找到实体，这些知识图指的是同一个真实世界的对象。KG嵌入的最新进展推动了基于嵌入的实体对齐技术的出现，该技术在连续嵌入空间中对实体进行编码，并基于学习到的嵌入来测量实体的相似性。本文对这一新兴领域进行了全面的实验研究。本研究调查了最近23种基于嵌入的实体对齐方法，并根据它们的技术和特点对它们进行了分类。我们进一步观察到，当前的方法在评估中使用不同的数据集，这些数据集中实体的度分布与实际KG不一致。因此，我们提出了一种新的KG抽样算法，通过该算法，我们生成了一组具有各种异质性和分布的专用基准数据集，用于现实评估。本研究还生成了一个开源库，其中包括12种具有代表性的基于嵌入的实体对齐方法。我们在生成的数据集上广泛评估这些方法，以了解它们的优点和局限性。此外，对于当前方法中尚未探索的几个方向，我们进行探索性实验，并报告我们的初步发现，以供未来研究。基准数据集、开源库和实验结果都可以在线访问，并将得到适当的维护。

*** **UPDATE** ***

- Aug. 1, 2021: We release the source code for [entity alignment with dangling cases](https://sunzequn.github.io/articles/acl2021_dbp2.pdf).

- June 29, 2021: We release the [DBP2.0](https://github.com/nju-websoft/OpenEA/tree/master/dbp2.0) dataset for [entity alignment with dangling cases](https://sunzequn.github.io/articles/acl2021_dbp2.pdf).

- Jan. 8, 2021: The results of AliNet on OpenEA datasets are avaliable at [Google docs](https://docs.google.com/spreadsheets/d/1P_MX8V7zOlZjhHlEMiXbXlIaMGSJT1Gh_gZWe4yIQBY/edit?usp=sharing).

- Nov. 30, 2020: We release **a new version (v2.0) of the OpenEA dataset**, where the URIs of DBpedia and YAGO entities are encoded to resovle the [name bias](https://www.aclweb.org/anthology/2020.emnlp-main.515.pdf) issue. It is strongly recommended to use the [v2.0 dataset](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3) for evaluating attribute-based entity alignment methods, such that the results can better reflect the robustness of these methods in real-world situation.

- Sep. 24, 2020: add AliNet.

## Table of contents
1. [基于嵌入的实体对齐](#library-for-embedding-based-entity-alignment)
    1. [概览](#overview)
    2. [上手](#getting-started)
        1. [代码包描述](#package-description)
        2. [依赖](#dependencies)
        3. [安装](#installation)
        4. [用法](#usage)
2. [KG 采样方法与数据](#kg-sampling-method-and-datasets)
    1. [基于度的迭代采样](#iterative-degree-based-sampling)
    2. [数据集概览](#dataset-overview)
    2. [数据集描述](#dataset-description)
3. [实验与结果](#experiment-and-results)
    1. [实验设置](#experiment-settings)
    2. [详细结果](#detailed-results)
4. [License](#license)
5. [Citation](#citation)

## 基于嵌入的实体对齐

### 概览

我们使用 [Python](https://www.python.org/) 与 [Tensorflow](https://www.tensorflow.org/) 开发了一个开源库，叫 **OpenEA**, 实现了基于嵌入的实体对齐。架构如下图所示：

<p>
  <img width="70%" src="https://cdn.jsdelivr.net/gh/nju-websoft/OpenEA/docs/stack.png" />
</p>

OpenEA 设计目标包括三个方面：松耦合，功能性与扩展性，现有解决方案。

* **松耦合**. 嵌入与对齐模块互相独立。OpenEA提供了一个带有预定义输入和输出数据结构的框架模板，使这三个模块成为一个完整的管道。用户可以在这些模块中自由调用和组合不同的技术。

* **功能性与扩展性**. OpenEA实现了一组必要的函数作为其底层组件，包括嵌入模块中的初始化函数、丢失函数和负采样方法；互动模式下的组合与学习策略；以及校准模块中的距离度量和校准推理策略。除此之外，OpenEA还提供了一组灵活的高级功能，以及调用底层组件的配置选项。通过这种方式，可以通过添加新的配置选项轻松集成新功能。

* **Off-the-shelf solutions**. 为了便于在不同的场景中使用OpenEA，我们尽最大努力集成或重建大多数现有的基于嵌入的实体对齐方法。目前，OpenEA集成了以下基于嵌入的实体对齐方法:
    1. **MTransE**: [Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment](https://www.ijcai.org/proceedings/2017/0209.pdf). IJCAI 2017.
    2. **IPTransE**: [Iterative Entity Alignment via Joint Knowledge Embeddings](https://www.ijcai.org/proceedings/2017/0595.pdf). IJCAI 2017.
    3. **JAPE**: [Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding](https://link.springer.com/chapter/10.1007/978-3-319-68288-4_37). ISWC 2017.
    4. **KDCoE**: [Co-training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment](https://www.ijcai.org/proceedings/2018/0556.pdf). IJCAI 2018.
    5. **BootEA**: [Bootstrapping Entity Alignment with Knowledge Graph Embedding](https://www.ijcai.org/proceedings/2018/0611.pdf). IJCAI 2018.
    6. **GCN-Align**: [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://www.aclweb.org/anthology/D18-1032). EMNLP 2018.
    7. **AttrE**: [Entity Alignment between Knowledge Graphs Using Attribute Embeddings](https://people.eng.unimelb.edu.au/jianzhongq/papers/AAAI2019_EntityAlignment.pdf). AAAI 2019.
    8. **IMUSE**: [Unsupervised Entity Alignment Using Attribute Triples and Relation Triples](https://link.springer.com/content/pdf/10.1007%2F978-3-030-18576-3_22.pdf). DASFAA 2019.
    9. **SEA**: [Semi-Supervised Entity Alignment via Knowledge Graph Embedding with Awareness of Degree Difference](https://dl.acm.org/citation.cfm?id=3313646). WWW 2019.
    10. **RSN4EA**: [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf). ICML 2019.
    11. **MultiKE**: [Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/0754.pdf). IJCAI 2019.
    12. **RDGCN**: [Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs](https://www.ijcai.org/proceedings/2019/0733.pdf). IJCAI 2019.
    13. **AliNet**: [Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation](https://aaai.org/ojs/index.php/AAAI/article/view/5354). AAAI 2020.
    
* OpenEA还在嵌入模块中集成了以下关系嵌入模型和两个属性嵌入模型（AC2Vec和Label2vec）:
    1. **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546). AAAI 2014.
    1. **TransR**: [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523). AAAI 2015.
    1. **TransD**: [Knowledge Graph Embedding via Dynamic Mapping Matrix](https://aclweb.org/anthology/P15-1067). ACL 2015.
    1. **HolE**: [Holographic Embeddings of Knowledge Graphs](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12484/11828). AAAI 2016.
    1. **ProjE**: [ProjE: Embedding Projection for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279/13906). AAAI 2017.
    1. **ConvE**: [Convolutional 2D Knowledge Graph Embeddings](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366/15884). AAAI 2018.
    1. **SimplE**: [SimplE Embedding for Link Prediction in Knowledge Graphs](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf). NeurIPS 2018.
    1. **RotatE**: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/pdf?id=HkgEQnRqYQ). ICLR 2019.

### 上手
这些说明包括如何获取库的副本，以及如何在本地计算机上安装和运行库，以用于开发和测试。它还概述了源代码的包结构。

#### 包描述

```
src/
├── openea/
│   ├── approaches/: 现有嵌入实体对齐模型实现
│   ├── models/: 未探索关系嵌入模型实现
│   ├── modules/: 嵌入模块、对齐模块及其交互框架的实现包
│   ├── expriment/: 评估方法的实现包
```

#### 依赖
* Python 3.x (tested on Python 3.6)
* Tensorflow 1.x (tested on Tensorflow 1.8 and 1.12)
* Scipy
* Numpy
* Graph-tool == 2.29 or igraph or NetworkX
* Pandas
* Scikit-learn
* Matching==0.1.1
* Gensim

#### 安装
我们建议创建一个新的conda环境来安装和运行OpenEA。您应该首先使用conda安装tensorflow gpu（在1.8和1.12上测试）、graph tool（在2.27和2.29上测试，最新版本会导致错误）和python igraph：

```bash
conda create -n openea python=3.6
conda activate openea
conda install tensorflow-gpu==1.12
conda install -c conda-forge graph-tool==2.29
conda install -c conda-forge python-igraph
```

然后使用以下步骤安装OpenEA：

```bash
git clone https://github.com/nju-websoft/OpenEA.git OpenEA
cd OpenEA
pip install -e .
```

#### 用法
下面是一个关于如何在Python中使用OpenEA的示例（我们假设您已经下载了我们的数据集，并按照[examples](https://github.com/nju-websoft/OpenEA/tree/master/run/args)中的说明配置了超参数）

```python
import openea as oa

model = oa.kge_model.TransE
args = load_args("hyperparameter file folder")
kgs = read_kgs_from_folder("data folder")
model.set_args(args)
model.set_kgs(kgs)
model.init()
model.run()
model.test()
model.save()

```
[更多示例](https://github.com/nju-websoft/OpenEA/tree/master/run)

要在我们的数据集上运行现成的方法并重现我们的实验，cd 进 ./run 目录并使用以下脚本：

```bash
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```

例如，如果要在D-W-15K（V1）的 first split 上运行BootEA，请执行以下脚本：

```bash
python main_from_args.py ./args/bootea_args_15K.json D_W_15K_V1 721_5fold/1/
```

## KG 采样方法和数据集

由于当前广泛使用的数据集与现实世界中的KG有很大不同，我们提出了一种新的数据集采样算法来生成基于嵌入的实体对齐的基准数据集。

### 基于度的迭代采样
提出的基于度的迭代采样（IDS）算法通过参考对齐同时删除两个源KG中的实体，直到达到所需的大小，同时保留采样数据集与源KG相似的度分布。下图描述了取样程序。

<p>
  <img width="50%" src="https://cdn.jsdelivr.net/gh/nju-websoft/OpenEA/docs/KG_sampling.png" />
</p>

### 数据集概览
 
我们选择了三个著名的KG作为我们的来源：DBpedia（2016-10）、Wikidata（20160801）和YAGO3。此外，我们考虑两种跨语言版本的DBPedia:英语-法语和英语-德语。我们遵循JAPE和BootEA中的约定，使用IDS算法生成两种大小的数据集，其中包含15K和100K个实体：

*#* Entities | Languages | Dataset names
:---: | :---: | :---: 
15K | Cross-lingual | EN-FR-15K, EN-DE-15K
15K | English | D-W-15K, D-Y-15K
100K | Cross-lingual | EN-FR-100K, EN-DE-100K
100K | English-lingual | D-W-100K, D-Y-100K

v1.1 数据集可以在此下载 [figshare](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/2), [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0) or [Baidu Wangpan](https://pan.baidu.com/s/1Wb4xMds3PT0IaKCJrPR8Lw) (password: 9feb). (**注意**, 我们在V1.0数据集的Yago中修复了一个小型格式问题。请从上面的链接下载我们的V1.1数据集，并使用此版本进行评估)

(**推荐？**) v2.0 数据集可以在此下载 [figshare](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3), [Dropbox](https://www.dropbox.com/s/xfehqm4pcd9yw0v/OpenEA_dataset_v2.0.zip?dl=0) or [Baidu Wangpan](https://pan.baidu.com/s/19RlM9OqwhIz4Lnogrp74tg) (password: nub1). 



### 数据集统计信息
 
我们为每对要对齐的KG生成两个版本的数据集。V1是直接使用IDS算法生成的。对于V2，我们首先随机删除源KG中度数较低（d<=5）的实体，使平均度数加倍，然后

然后执行IDS以适应新的KG。数据集的统计数据如下所示。

<p>
  <img src="https://cdn.jsdelivr.net/gh/nju-websoft/OpenEA/docs/Dataset_Statistics.png" />
</p>

### 数据集介绍
We hereby take the EN_FR_15K_V1 dataset as an example to introduce the files in each dataset. In the *721_5fold* folder, we divide the reference entity alignment into five disjoint folds, each of which accounts for 20% of the total alignment. For each fold, we pick this fold (20%) as training data and leave the remaining (80%) for validation (10%) and testing (70%). The directory structure of each dataset is listed as follows:
在此，我们以EN_FR_15K_V1数据集为例，介绍每个数据集中的文件。在*721_5fold*文件夹中，我们将参考实体对齐划分为五个不相交的fold，每个fold占总路线的20%。对于每个fold，我们选择这个fold（20%）作为训练数据，剩下的（80%）用于验证（10%）和测试（70%）。每个数据集的目录结构如下所示：

```
EN_FR_15K_V1/
├── attr_triples_1: attribute triples in KG1
├── attr_triples_2: attribute triples in KG2
├── rel_triples_1: relation triples in KG1
├── rel_triples_2: relation triples in KG2
├── ent_links: entity alignment between KG1 and KG2
├── 721_5fold/: entity alignment with test/train/valid (7:2:1) splits
│   ├── 1/: the first fold
│   │   ├── test_links
│   │   ├── train_links
│   │   └── valid_links
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
```

## 实验及结果

### 实验设置
OpenEA使用的常见超参数如下所示。

<table style="text-align:center">
    <tr>
        <td style="text-align:center"></td>
        <th style="text-align:center">15K</th>
        <th style="text-align:center">100K</th>
    </tr>
    <tr>
        <td style="text-align:center">Batch size for rel. triples</td>
        <td style="text-align:center">5,000</td>
        <td style="text-align:center">20,000</td>
    </tr>
    <tr>
        <td style="text-align:center">终止条件 (Termination condition)</td>
        <td style="text-align:center" colspan="2">Early stop when the Hits@1 score begins to drop on <br>
            the validation sets, checked every 10 epochs.</td>
    </tr>
    <tr>
        <td style="text-align:center">Max. epochs</td>
        <td style="text-align:center" colspan="2">2,000</td>
    </tr>
</table>

此外，将数据集分成培训，验证和测试集，得到了很好的认可
细节如下

| *#* Ref. alignment | *#* Training | *#* Validation | *#* Test |
|:------------------:|:------------:|:--------------:|:--------:|
|        15K         |    3,000     |     1,500      |  10,500  |
|        100K        |    20,000    |     10,000     |  70,000  |

我们使用 Hits@m (m = 1, 5, 10, 50), mean rank (MR) and mean reciprocal rank (MRR) 作为评估指标。 更高的 Hits@m 和 MRR scores 以及 更低的 MR 分数意味着更好的性能。

### Detailed Results
The detailed and supplementary experimental results are list as follows:

#### Detailed results of current approaches on the 15K datasets
[**detailed_results_current_approaches_15K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_current_approaches_15K.csv)

#### Detailed results of current approaches on the 100K datasets
[**detailed_results_current_approaches_100K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_current_approaches_100K.csv)

#### Running time (sec.) of current approaches
[**running_time.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/running_time.csv)

### Unexplored KG Embedding Models

#### Detailed results of unexplored KG embedding models on the 15K datasets
[**detailed_results_unexplored_models_15K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_unexplored_models_15K.csv)

#### Detailed results of unexplored KG embedding models on the 100K datasets
[**detailed_results_unexplored_models_100K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_unexplored_models_100K.csv)

## License
This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## Citation
If you find the benchmark datasets, the OpenEA library or the experimental results useful, please kindly cite the following paper:
```
@article{OpenEA,
  author    = {Zequn Sun and
               Qingheng Zhang and
               Wei Hu and
               Chengming Wang and
               Muhao Chen and
               Farahnaz Akrami and
               Chengkai Li},
  title     = {A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {13},
  number    = {11},
  pages     = {2326--2340},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2326-sun.pdf}
}
```

If you use the DBP2.0 dataset, please kindly cite the following paper:
```
@inproceedings{DBP2,
  author    = {Zequn Sun and
               Muhao Chen and
               Wei Hu},
  title     = {Knowing the No-match: Entity Alignment with Dangling Cases},
  booktitle = {ACL},
  year      = {2021}
}
```

