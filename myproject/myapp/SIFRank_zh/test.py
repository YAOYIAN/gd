#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac
import logging
logging.basicConfig(level=logging.ERROR)

#download from https://github.com/HIT-SCIR/ELMoForManyLangs
model_file = r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/zhs.model/'

ELMO = word_emb_elmo.WordEmbeddings(model_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
#download from http://thulac.thunlp.org/
zh_model = thulac.thulac(model_path=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/thulac.models/',user_dict=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/user_dict.txt')
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "计算机科学与技术（Computer Science and Technology）是国家一级学科，下设信息安全、软件工程、计算机软件与理论、计算机系统结构、计算机应用技术、计算机技术等专业。 [1]主修大数据技术导论、数据采集与处理实践（Python）、Web前/后端开发、统计与数据分析、机器学习、高级数据库系统、数据可视化、云计算技术、人工智能、自然语言处理、媒体大数据案例分析、网络空间安全、计算机网络、数据结构、软件工程、操作系统等课程，以及大数据方向系列实验，并完成程序设计、数据分析、机器学习、数据可视化、大数据综合应用实践、专业实训和毕业设计等多种实践环节。"
text2 = "静电学是研究静止电荷产生电场及电场对电荷作用规律的学科。电荷只有两种，称为正电和负电。同种电荷相互排斥，异种电荷相互吸引。电荷遵从电荷守恒定律。电荷可以从一个物体转移到另一个物体，任何物理过程中电荷的代数和保持不变。所谓带电，不过是正负电荷的分离或转移；所谓电荷消失，不过是正负电荷的中和。静止电荷之间相互作用力符合库仑定律：在真空中两个静止点电荷之间作用力的大小与它们的电荷量的乘积成正比，与它们之间的距离的平方成反比；作用力的方向沿着它们之间的联线，同号电荷相斥，异号电荷相吸。电荷之间相互作用力是通过电荷产生的电场相互作用的。电荷产生的电场用电场强度(简称场强)来描述。空间某一点的电场强度用正的单位试探电荷在该点所受的电场力来定义，电场强度遵从场强叠加原理。通常的物质，按其导电性能的不同可分两种情况：导体和绝缘体。导体体内存在可运动的自由电荷；绝缘体又称为电介质，体内只有束缚电荷。在电场的作用下，导体内的自由电荷将产生移动。当导体的成分和温度均匀时，达到静电平衡的条件是导体内部的电场强度处处等于零。根据这一条件，可导出导体静电平衡的若干性质。静磁学是研究电流稳恒时产生磁场以及磁场对电流作用力的学科。电荷的定向流动形成电流。电流之间存在磁的相互作用，这种磁相互作用是通过磁场传递的，即电流在其周围的空间产生磁场，磁场对放置其中的电流施以作用力。电流产生的磁场用磁感应强度描述。电磁场是研究随时间变化下的电磁现象和规律的学科。当穿过闭导体线圈的磁通量发生变化时，线圈上产生感应电流。感应电流的方向可由楞次定律确定。闭合线圈中的感应电流是感应电动势推动的结果，感应电动势遵从法拉第定律：闭合线圈上的感应电动势的大小总是与穿过线圈的磁通量的时间变化率成正比。麦克斯韦方程组描述了电磁场普遍遵从的规律。它同物质的介质方程、洛仑兹力公式以及电荷守恒定律结合起来，原则上可以解决各种宏观电动力学问题。根据麦克斯韦方程组导出的一个重要结果是存在电磁波，变化的电磁场以电磁波的形式传播，电磁波在真空中的传播速度等于光速。这也说明光也是电磁波的一种，因此光的波动理论纳入了电磁理论的范畴。"
keyphrases = SIFRank(text2, SIF, zh_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text2, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
# print("------------------------------------------")
# print("原文:"+text)
print("------------------------------------------")
print("SIFRank_zh结果:")
print(keyphrases)
print("SIFRank+_zh结果:")
print(keyphrases_)

keyphrases = SIFRank(text, SIF, zh_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
# print("------------------------------------------")
# print("原文:"+text)
print("------------------------------------------")
print("SIFRank_zh结果:")
print(keyphrases)
print("SIFRank+_zh结果:")
print(keyphrases_)
# print("------------------------------------------")
# print("jieba分词TFIDF算法结果:")
# print(jieba.analyse.tfidf(text, topK=15, withWeight=True, allowPOS=('n','nr','ns')))
# print("jieba分词TFIDF算法结果:")
# print(jieba.analyse.textrank(text, topK=15, withWeight=True, allowPOS=('n','nr','ns')))