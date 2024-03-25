from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac

#download from https://github.com/HIT-SCIR/ELMoForManyLangs
model_file = r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/zhs.model/'

ELMO = word_emb_elmo.WordEmbeddings(model_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
#download from http://thulac.thunlp.org/
zh_model = thulac.thulac(model_path=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/thulac.models/',user_dict=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/user_dict.txt')
elmo_layers_weight = [0.5, 0.5, 0.0]

text = "计算机科学与技术(Computer Science and Technology)是国家一级学科，下设信息安全、软件工程、计算机软件与理论、计算机系统结构、计算机应用技术、计算机技术等专业。 [1]主修大数据技术导论、数据采集与处理实践（Python）、Web前/后端开发、统计与数据分析、机器学习、高级数据库系统、数据可视化、云计算技术、人工智能、自然语言处理、媒体大数据案例分析、网络空间安全、计算机网络、数据结构、软件工程、操作系统等课程，以及大数据方向系列实验，并完成程序设计、数据分析、机器学习、数据可视化、大数据综合应用实践、专业实训和毕业设计等多种实践环节。"
keyphrases = SIFRank(text, SIF, zh_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)