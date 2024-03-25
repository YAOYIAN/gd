# from django.http import HttpResponse
import shutil
import zipfile
from django.http import HttpResponse, FileResponse
from django.shortcuts import render
import os
from pke.supervised import WINGNUS
from random import random
import nltk
from summarizer.util import AGGREGATE_MAP
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
import torch
from spacy.lang.en import English
from spacy.language import Language
from typing import Dict, List, Union, Callable, Tuple, Optional
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
import torch
from pylatex import Document, PageStyle, Head, MiniPage, Foot, LargeText, \
    MediumText, LineBreak, simple_page_number, Section, Subsection, Tabular, Command, Package
from pylatex.utils import bold, italic, NoEscape
import numpy as np
import torch.nn.functional as F
from pylatex import Document, Section, Command
import requests
import random
from hashlib import md5
from . import models
from .SIFRank_zh.embeddings import sent_emb_sif, word_emb_elmo
from .SIFRank_zh.model.method import SIFRank
import thulac
import logging
import re
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
logging.basicConfig(level=logging.ERROR)
nlp = spacy.load("en_core_web_sm")
LOWER = False
tokenizer = AutoTokenizer.from_pretrained('G:/Graduation_Design/dataset/all-mpnet-base-v2')
model = AutoModel.from_pretrained('G:/Graduation_Design/dataset/all-mpnet-base-v2')
print("all-mpnet-base-v2 load over")
model_brio = BartForConditionalGeneration.from_pretrained(
    'G:/Graduation_Design/dataset/BRIO-main/BRIO-main/cnndm_cased_model')
tokenizer_brio = BartTokenizer.from_pretrained('G:/Graduation_Design/dataset/BRIO-main/BRIO-main/cnndm_cased_model')
print("BRIO load over")
tokenizer_titlegen = AutoTokenizer.from_pretrained(
    "G:/Graduation_Design/dataset/BRIO-main/BRIO-main/autonlp_sci_titlemodel")
model_titlegen = AutoModelForSeq2SeqLM.from_pretrained(
    "G:/Graduation_Design/dataset/BRIO-main/BRIO-main/autonlp_sci_titlemodel")
print("autonlp_sci_titlemodel load over")
tokenizer_longsumm = AutoTokenizer.from_pretrained("G:/Graduation_Design/dataset/HipoRank-master/HipoRank-master/lsg-bart-base-16384-arxiv", trust_remote_code=True,revision="")
model_longsumm = AutoModelForSeq2SeqLM.from_pretrained("G:/Graduation_Design/dataset/HipoRank-master/HipoRank-master/lsg-bart-base-16384-arxiv", trust_remote_code=True,revision="")
pipe = pipeline("text2text-generation", model=model_longsumm, tokenizer=tokenizer_longsumm, device=-1)
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)
infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer
print("longsumm load over")




def translate(fromlang,deslang,text):
    appid = '20231212001908494'
    appkey = 'DICIBsCMWGgGzDlEmhWV'
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = fromlang
    to_lang =  deslang
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    query = text
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    return result["trans_result"][0]["dst"]

def split_string_by_keywords(string, keywords):
    pattern = '|'.join(keywords)
    segments = re.findall(pattern, string)
    result = []
    start = 0
    for segment in segments:
        end = string.index(segment, start)
        result.append((string[start:end], False))
        result.append((segment, True))
        start = end + len(segment)
    result.append((string[start:], False))
    return result

def generate_header(doc):
    # Add document header
    header = PageStyle("header")
    # Create left header
    with header.create(Head("L")):
        header.append("Date: ")
        header.append(NoEscape(r'\today'))

    # Create center header
    with header.create(Head("C")):
        header.append("Beijing Normal University")
    # Create right header
    with header.create(Head("R")):
        header.append(simple_page_number())
    # Create left footer
    with header.create(Foot("L")):
        header.append("Left Footer")
    # Create center footer
    with header.create(Foot("C")):
        header.append("Center Footer")
    # Create right footer
    with header.create(Foot("R")):
        header.append("Right Footer")

    doc.preamble.append(header)
    doc.change_document_style("header")

def split_into_sentences_struct(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def cosine_sim_struct(c1, c2):
    try:
        # works for Counter
        n1 = np.sqrt(sum([x * x for x in list(c1.values())]))
        n2 = np.sqrt(sum([x * x for x in list(c2.values())]))
        num = sum([c1[key] * c2[key] for key in c1])
    except:
        # works for ordinary list
        assert (len(c1) == len(c2))
        n1 = np.sqrt(sum([x * x for x in c1]))
        n2 = np.sqrt(sum([x * x for x in c2]))
        num = sum([c1[i] * c2[i] for i in range(len(c1))])
    try:
        if n1 * n2 < 1e-9:  # divide by zero case
            return 0
        return num / (n1 * n2)
    except:
        return 0

class EnglishTokenizer_struct:
    def __init__(self):
        pass

    def tokenize_struct(self, text):
        return text.lower().split()


class C99_struct:
    def __init__(self, window=4, std_coeff=1.0, tokenizer=EnglishTokenizer_struct()):
        self.window = window
        self.sim = None
        self.rank = None
        self.sm = None
        self.std_coeff = std_coeff
        self.tokenizer = tokenizer

    def segment_struct(self, document):
        if len(document) < 3:
            return [1] + [0 for _ in range(len(document) - 1)]
        n = len(document)
        self.window = min(self.window, n)
        cnts = document
        self.sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                self.sim[i][j] = cosine_sim_struct(cnts[i], cnts[j])
                self.sim[j][i] = self.sim[i][j]

        self.rank = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r1 = max(0, i - self.window + 1)
                r2 = min(n - 1, i + self.window - 1)
                c1 = max(0, j - self.window + 1)
                c2 = min(n - 1, j + self.window - 1)
                sublist = self.sim[r1:(r2 + 1), c1:(c2 + 1)].flatten()
                lowlist = [x for x in sublist if x < self.sim[i][j]]
                self.rank[i][j] = 1.0 * len(lowlist) / ((r2 - r1 + 1) * (c2 - c1 + 1))
                self.rank[j][i] = self.rank[i][j]

        self.sm = np.zeros((n, n))
        prefix_sm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                prefix_sm[i][j] = self.rank[i][j]
                if i - 1 >= 0: prefix_sm[i][j] += prefix_sm[i - 1][j]
                if j - 1 >= 0: prefix_sm[i][j] += prefix_sm[i][j - 1]
                if i - 1 >= 0 and j - 1 >= 0: prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
        for i in range(n):
            for j in range(i, n):
                if i == 0:
                    self.sm[i][j] = prefix_sm[j][j]
                else:
                    self.sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
                                    - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
                self.sm[j][i] = self.sm[i][j]

        D = 1.0 * self.sm[0][n - 1] / (n * n)
        darr, region_arr, idx = [D], [Region_struct(0, n - 1, self.sm)], []
        sum_region, sum_area = float(self.sm[0][n - 1]), float(n * n)
        for i in range(n - 1):
            mx, pos = -1e9, -1
            for j, region in enumerate(region_arr):
                if region.l == region.r:
                    continue
                region.split_struct(self.sm)
                den = sum_area - region.area + region.lch.area + region.rch.area
                cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
                if cur > mx:
                    mx, pos = cur, j
            assert (pos >= 0)
            tmp = region_arr[pos]
            region_arr[pos] = tmp.rch
            region_arr.insert(pos, tmp.lch)
            sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
            sum_area += tmp.lch.area + tmp.rch.area - tmp.area
            darr.append(sum_region / sum_area)
            idx.append(tmp.best_pos)
        dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]
        smooth_dgrad = [dgrad[i] for i in range(len(dgrad))]
        if len(dgrad) > 1:
            smooth_dgrad[0] = (dgrad[0] * 2 + dgrad[1]) / 3.0
            smooth_dgrad[-1] = (dgrad[-1] * 2 + dgrad[-2]) / 3.0
        for i in range(1, len(dgrad) - 1):
            smooth_dgrad[i] = (dgrad[i - 1] + 2 * dgrad[i] + dgrad[i + 1]) / 4.0
        dgrad = smooth_dgrad

        avg, stdev = np.average(dgrad), np.std(dgrad)
        cutoff = avg + self.std_coeff * stdev
        assert (len(idx) == len(dgrad))
        above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
        if len(above_cutoff_idx) == 0:
            boundary = []
        else:
            boundary = idx[:max(above_cutoff_idx) + 1]
        ret = [0 for _ in range(n)]
        for i in boundary:
            ret[i] = 1
            for j in range(i - 1, i + 2):
                if 0 <= j < n and j != i and ret[j] == 1:
                    ret[i] = 0
                    break
        return [1] + ret[:-1]

class Region_struct:
    def __init__(self, l, r, sm_matrix):
        assert (r >= l)
        self.tot = sm_matrix[l][r]
        self.l = l
        self.r = r
        self.area = (r - l + 1) ** 2
        self.lch, self.rch, self.best_pos = None, None, -1

    def split_struct(self, sm_matrix):
        if self.best_pos >= 0:
            return
        if self.l == self.r:
            self.best_pos = self.l
            return
        assert (self.r > self.l)
        mx, pos = -1e9, -1
        for i in range(self.l, self.r):
            carea = (i - self.l + 1) ** 2 + (self.r - i) ** 2
            cur = (sm_matrix[self.l][i] + sm_matrix[i + 1][self.r]) / carea
            if cur > mx:
                mx, pos = cur, i
        assert (pos >= self.l and pos < self.r)
        self.lch = Region_struct(self.l, pos, sm_matrix)
        self.rch = Region_struct(pos + 1, self.r, sm_matrix)
        self.best_pos = pos

def mean_pooling_struct(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_convs_struct(embedings_list):
    model = C99_struct(window=12, std_coeff=1)
    sent_label = []
    data = embedings_list
    for i in range(0, len(data)):
        boundary = model.segment_struct(data[i])
        temp_labels = []
        l = 0
        for j in range(0, len(boundary)):
            if boundary[j] == 1:
                l += 1
            temp_labels.append(l)
        sent_label.append(temp_labels)
    return sent_label

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SBertEmbedding:
    def __init__(self, tokenizer,model):
        self.sbert_model = model
        self.sbert_tokenizer = tokenizer
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert_model.to(self.device)

    def extract_embeddings(self, sentences: List[str]) -> np.ndarray:
        encoded_input = self.sbert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input
        with torch.no_grad():
            model_output = self.sbert_model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings = sentence_embeddings
        return embeddings

    def __call__(self, sentences: List[str]) -> np.ndarray:
        return self.extract_embeddings(sentences)

class SentenceABC:
    def __init__(self, nlp: Language, is_spacy_3: bool):
        self.nlp = nlp
        self.is_spacy_3 = is_spacy_3

    def sentence_processor(
            self, doc, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        to_return = []
        for c in doc.sents:
            if max_length > len(c.text.strip()) > min_length:
                if self.is_spacy_3:
                    to_return.append(c.text.strip())
                else:
                    to_return.append(c.string.strip())
        return to_return

    def process(
            self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        raise NotImplementedError()

    def __call__(
            self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        return self.process(body, min_length, max_length)


class SentenceHandler(SentenceABC):
    def __init__(self, language: Language = English):
        nlp = language()
        is_spacy_3 = False
        try:
            # Supports spacy 2.0
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        except Exception:
            # Supports spacy 3.0
            nlp.add_pipe("sentencizer")
            is_spacy_3 = True

        super().__init__(nlp, is_spacy_3)

    def process(
            self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        doc = self.nlp(body)
        return self.sentence_processor(doc, min_length, max_length)

class SummaryProcessor:
    def __init__(
            self,
            model: Callable,
            sentence_handler: SentenceHandler,
            random_state: int = 12345
    ):
        np.random.seed(random_state)
        self.model = model
        self.sentence_handler = sentence_handler
        self.random_state = random_state

    def calculate_elbow(
            self,
            body: str,
            algorithm: str = 'kmeans',
            min_length: int = 40,
            max_length: int = 600,
            k_max: int = None,
    ) -> List[float]:
        sentences = self.sentence_handler(body, min_length, max_length)
        if k_max is None:
            k_max = len(sentences) - 1
        hidden = self.model(sentences)
        elbow = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_elbow(k_max)
        return elbow

    def calculate_optimal_k(
            self,
            body: str,
            algorithm: str = 'kmeans',
            min_length: int = 40,
            max_length: int = 600,
            k_max: int = None,
    ) -> int:
        sentences = self.sentence_handler(body, min_length, max_length)
        if k_max is None:
            k_max = len(sentences) - 1
        hidden = self.model(sentences)
        optimal_k = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_optimal_cluster(k_max)
        return optimal_k

    def cluster_runner(
            self,
            sentences: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = 3,
    ) -> Tuple[List[str], np.ndarray]:
        first_embedding = None
        hidden = self.model(sentences)
        if use_first:
            num_sentences = num_sentences - 1 if num_sentences else num_sentences
            if len(sentences) <= 1:
                return sentences, hidden
            first_embedding = hidden[0, :]
            hidden = hidden[1:, :]
        summary_sentence_indices = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).cluster(ratio, num_sentences)
        if use_first:
            if summary_sentence_indices:
                summary_sentence_indices = [i + 1 for i in summary_sentence_indices]
                summary_sentence_indices.insert(0, 0)
            else:
                summary_sentence_indices.append(0)
            hidden = np.vstack([first_embedding, hidden])
        sentences = [sentences[j] for j in summary_sentence_indices]
        embeddings = np.asarray([hidden[j] for j in summary_sentence_indices])
        return sentences, embeddings

    def run_embeddings(
            self,
            body: str,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            aggregate: str = None,
    ) -> Optional[np.ndarray]:
        sentences = self.sentence_handler(body, min_length, max_length)
        if sentences:
            _, embeddings = self.cluster_runner(sentences, ratio, algorithm, use_first, num_sentences)
            if aggregate is not None:
                assert aggregate in [
                    'mean', 'median', 'max', 'min'], "aggregate must be mean, min, max, or median"
                embeddings = AGGREGATE_MAP[aggregate](embeddings, axis=0)
            return embeddings
        return None

    def run(
            self,
            body: str,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False,
    ) -> Union[List, str]:
        sentences = self.sentence_handler(body, min_length, max_length)
        if sentences:
            sentences, _ = self.cluster_runner(sentences, ratio, algorithm, use_first, num_sentences)
        if return_as_list:
            return sentences
        else:
            return ' '.join(sentences)

    def __call__(
            self,
            body: str,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False,
    ) -> str:
        return self.run(body, ratio, min_length, max_length,
                        use_first, algorithm, num_sentences, return_as_list)

class SBertSummarizer(SummaryProcessor):
    def __init__(
            self,
            sentence_handler: SentenceHandler = SentenceHandler(),
            random_state: int = 12345
    ):
        model_func = SBertEmbedding(tokenizer,model)
        super().__init__(
            model=model_func, sentence_handler=sentence_handler, random_state=random_state
        )

class ClusterFeatures:
    def __init__(
            self,
            features: ndarray,
            algorithm: str = 'kmeans',
            pca_k: int = None,
            random_state: int = 12345,
    ):
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features
        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state

    def _get_model(self, k: int) -> Union[GaussianMixture, KMeans]:
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state)

    def _get_centroids(self, model: Union[GaussianMixture, KMeans]) -> np.ndarray:
        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids: np.ndarray) -> Dict:
        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []
        for j, centroid in enumerate(centroids):
            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)
                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value
            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1
        return args

    def calculate_elbow(self, k_max: int) -> List[float]:
        inertias = []
        for k in range(1, min(k_max, len(self.features))):
            model = self._get_model(k).fit(self.features)
            inertias.append(model.inertia_)
        return inertias

    def calculate_optimal_cluster(self, k_max: int) -> int:
        delta_1 = []
        delta_2 = []
        max_strength = 0
        k = 1
        inertias = self.calculate_elbow(k_max)
        for i in range(len(inertias)):
            delta_1.append(inertias[i] - inertias[i - 1] if i > 0 else 0.0)
            delta_2.append(delta_1[i] - delta_1[i - 1] if i > 1 else 0.0)
        for j in range(len(inertias)):
            strength = 0 if j <= 1 or j == len(inertias) - 1 else delta_2[j + 1] - delta_1[j + 1]
            if strength > max_strength:
                max_strength = strength
                k = j + 1
        return k

    def cluster(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        if num_sentences is not None:
            if num_sentences == 0:
                return []
            k = min(num_sentences, len(self.features))
        else:
            k = max(int(len(self.features) * ratio), 1)
        model = self._get_model(k).fit(self.features)
        centroids = self._get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)
        sorted_values = sorted(cluster_args.values())
        return sorted_values

    def __call__(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        return self.cluster(ratio)
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


from django.http import JsonResponse


def your_view(request):
    data_1 = request.GET.get('key1')
    with open(data_1+"/state.txt",'r') as f:
        data = f.readline()
    # 处理前端Ajax请求
    print(data)
    # data  # 要返回给前端的数据
    return JsonResponse(data, safe=False)
def index(request):
    context = {
        'myapp_list': [
            {
                "title": "输出包含如下文件：",
                "content_1": "您可以直接使用生成的ppt.pdf和note.pdf，如需更改也可以通过修改.tex文件，然后再自行编译。",
                "content_2": " ",
            },
            {
                "title": "1. PPT部分：",
                "content_1": "1.1 ppt.pdf",
                "content_2": "1.2 ppt.tex",
            },
            {
                "title": "2. NOTE部分：",
                "content_1": "2.1 note.pdf",
                "content_2": "2.2 note.tex",
            },
        ]
    }
    return render(request, 'myapp/index.html', context=context)

def download(request):
    user_input = request.POST.get('text_field_1')
    user_input_last = request.POST.get('text_field_last')
    user_ip = request.POST.get('hidden_ip')
    print("user_ip: " + user_ip)
    process_input2_4files(user_input,user_input_last,user_ip)
    zipDir(user_ip+'/prezip', user_ip+'/result.zip')
    file_path = user_ip+'/result.zip'  # 文件的路径
    if os.path.exists(file_path):
        file = open(file_path, 'rb')
        response = FileResponse(file, content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(file_path)
        return response
    else:
        return HttpResponse('File not found')

def zipDir(dirpath, outFullName):
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def process_input2_4files(input_passage,user_input_last,user_ip):
    with open(user_ip + "/state.txt",'w') as f:
        f.write("已经开始处理")
    folder_path = user_ip+'/prezip'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    appid_fl = '20231212001908494'
    appkey_fl = 'DICIBsCMWGgGzDlEmhWV'
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/language'
    url = endpoint + path
    query = input_passage[:26]
    salt = random.randint(32768, 65536)
    sign = make_md5(appid_fl + query + str(salt) + appkey_fl)
    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid_fl, 'q': query, 'salt': salt, 'sign': sign}
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result_fl = r.json()
    with open(user_ip + "/state.txt",'w') as f:
        f.write("input language is "+result_fl["data"]["src"])
    print("input language is " + result_fl["data"]["src"])
    if result_fl["data"]["src"] == "en":
        input_article = input_passage
    elif result_fl["data"]["src"] == "zh":
        input_article = translate('zh', 'en', input_passage)
        # download from https://github.com/HIT-SCIR/ELMoForManyLangs
        model_file = r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/zhs.model/'
        ELMO = word_emb_elmo.WordEmbeddings(model_file)
        SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
        # download from http://thulac.thunlp.org/
        zh_model = thulac.thulac(
            model_path=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/thulac.models/',
            user_dict=r'G:/Graduation_Design/dataset/SIFRank_zh-master/SIFRank_zh-master/auxiliary_data/user_dict.txt')
        elmo_layers_weight = [0.0, 1.0, 0.0]
    with open(user_ip + "/state.txt",'w') as f:
        f.write("Full text summary in progress...")
    generated_text = pipe(input_article, truncation=True, max_length=16384, no_repeat_ngram_size=7, num_beams=2,
                              early_stopping=True)
    generated_text = generated_text[0]["generated_text"].replace('\n', ' ').strip()
    print("all summ complete")
  #  =====================================
    if result_fl["data"]["src"] == 'zh':
        user_input_last = translate('zh', 'en', user_input_last)
    inputs_last = tokenizer_brio([user_input_last], max_length=1024, return_tensors="pt", truncation=True)
    # Generate Summary
    summ_last = tokenizer_brio.batch_decode(model_brio.generate(inputs_last["input_ids"]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if result_fl["data"]["src"] == 'zh':
        summ_last = translate('en', 'zh', summ_last)
    summ_last = summ_last.replace('\n', ' ').strip()
    #=================


    with open(user_ip + "/state.txt",'w') as f:
        f.write("Full text summary over")


    # start split:
    sentences_list = split_into_sentences_struct(input_article)
    sentences_list = [sen.strip() for sen in sentences_list]
    sentences_1 = sentences_list
    encoded_input_1 = tokenizer(sentences_1, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with open(user_ip + "/state.txt",'w') as f:
        f.write("Article segmentation in progress...")
    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
    # Perform pooling
    sentence_embeddings_1 = mean_pooling_struct(model_output_1, encoded_input_1['attention_mask'])
    # Normalize embeddings
    embedings_list = F.normalize(sentence_embeddings_1, p=2, dim=1)
    embedings_list = [embedings_list.detach().numpy()]
    l = encode_convs_struct(embedings_list)
    max_num = 0
    struct_ans = []
    for index in range(len(l[0])):
        if max_num < l[0][index]:
            struct_ans.append("))))))))))")
            max_num += 1
        struct_ans.append(str(sentences_list[index]))

    struct_list = []
    string_add = ""
    is_first = True
    for item in struct_ans:
        if item == "))))))))))":
            if is_first:
                is_first = False
            else:
                struct_list.append(string_add)
            string_add = ""
        else:
            string_add += item
            string_add += "\n"
    struct_list.append(string_add)


    show_summ_struct = ""
    show_summ_struct += generated_text
    show_summ_struct += "\n========above all summ========\n"
    summs = []
    titles = []
    with open(user_ip + "/state.txt",'w') as f:
        f.write("Article segmentation is over")
    numm = 0
    for input_value in struct_list:
        numm += 1
        with open(user_ip + "/state.txt", 'w') as f:
            f.write("Process segmentation：{}/{}".format(numm, len(struct_list)))
        print(1)
        max_length = 1024
        # generation example test git
        article = input_value
        inputs = tokenizer_brio([article], max_length=max_length, return_tensors="pt", truncation=True)
        # Generate Summary
        summary_ids = model_brio.generate(inputs["input_ids"])
        sec_summ = tokenizer_brio.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        inputs = tokenizer_titlegen([str(sec_summ)], return_tensors="pt")
        summary_ids = model_titlegen.generate(inputs["input_ids"])

        title = tokenizer_titlegen.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace('\n', ' ').strip()
        show_summ_struct += title
        titles.append(title)
        show_summ_struct += "\n--------above title----------\n"

        show_summ_struct += sec_summ
        summs.append(sec_summ)
        show_summ_struct += "\n-------------------------\n"

    print("information")
    print(show_summ_struct)
    with open(user_ip + "/state.txt", 'w') as f:
        f.write("In the process of document/ppt writing...")
    geometry_options = {"margin": "0.7in"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.add(Package('ctex'))
    generate_header(doc)
    with doc.create(MiniPage(align='c')):
        doc.append(LargeText(bold("Summary of the article")))
        doc.append(LineBreak())
        doc.append(MediumText(italic("Automatically generated")))
    ppt = Document(user_ip+'/prezip/ppt', documentclass='beamer')
    ppt.packages.add(Package('ctex'))
    ppt.preamble.append(Command('usetheme', "Madrid"))  # Madrid, CambridgeUS
    ppt.preamble.append(Command('usecolortheme', "beaver"))  # beaver, default
    ppt.preamble.append(Command('usepackage', "blkarray"))
    ppt.append(Command('title', 'Slides for Lecture Notes'))
    ppt.append(Command('subtitle', Command('textit', 'Automatically generated')))
    ppt.append(Command('author', 'Yian Yao'))
    ppt.append(Command('date', NoEscape(r'\today')))
    ppt.append(NoEscape(r'\maketitle'))
    ppt.append(Command('begin', 'frame'))
    ppt.append(Command("frametitle", "Table of Contents"))
    ppt.append(Command('tableofcontents'))
    ppt.append(Command('end', 'frame'))

    if result_fl["data"]["src"] == 'zh':
        generated_text = translate('en', 'zh', generated_text)
    generated_text_bring_to_last = generated_text


    #========================
    if result_fl["data"]["src"] == 'zh':
        with doc.create(Section("首先回顾一下上节课的知识点")):
            keyphrases = SIFRank(summ_last, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
            keywords = [i[0] for i in keyphrases]
            result = split_string_by_keywords(summ_last, keywords)
            for tu in result:
                if tu[1] == True:
                    doc.append(bold(italic(tu[0])))
                else:
                    doc.append(italic(tu[0]))
            doc.append("\n")
            doc.append("**接下来我们学习几个重要的知识点。**")
            ppt.append(Command('section', '回顾上一节课的知识点'))
            ppt.append(Command('subsection', '这里是关于上次课程的回顾'))
    elif result_fl["data"]["src"] == 'en':
        with doc.create(Section("Let's review the knowledge points from the previous class")):
            nnnn = nlp(summ_last)
            extractor = WINGNUS()
            extractor.load_document(input=nnnn, language='en')
            extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
            extractor.candidate_weighting()
            keywords = [u for u, v in extractor.get_n_best(n=6, stemming=True)]

            result = split_string_by_keywords(summ_last, keywords)
            for tu in result:
                if tu[1] == True:
                    doc.append(bold(italic(tu[0])))
                else:
                    doc.append(italic(tu[0]))
            doc.append("\n")
            doc.append("**Next, we will learn several important knowledge points.**")

            ppt.append(Command('section', 'Last Course Review'))
            ppt.append(Command('subsection', 'Here is a review of the last course'))

    #=========================


    generated_text = summ_last
    if result_fl["data"]["src"] == 'zh':
        generated_text_list = generated_text.split('。')
    elif result_fl["data"]["src"] == 'en':
        generated_text_list = generated_text.split('.')
    print(generated_text_list)

    some_allsumm_text = []
    count_sum = 0
    temp_summ = ""
    for summ_sentence in generated_text_list:
        if result_fl["data"]["src"] == 'en':
            if len(summ_sentence) + count_sum > 800:
                some_allsumm_text.append(temp_summ)
                temp_summ = summ_sentence + '.'
                count_sum = len(summ_sentence)
            else:
                count_sum += len(summ_sentence)
                temp_summ += summ_sentence + '.'
        elif result_fl["data"]["src"] == 'zh':
            if len(summ_sentence) + count_sum > 300:
                some_allsumm_text.append(temp_summ)
                temp_summ = summ_sentence + '。'
                count_sum = len(summ_sentence)
            else:
                count_sum += len(summ_sentence)
                temp_summ += summ_sentence + '。'
    some_allsumm_text.append(temp_summ)

    for show_allsumm_sentence in some_allsumm_text:
        if result_fl["data"]["src"] == 'en':
            ppt.append(Command('begin', 'frame'))
            ppt.append(Command("frametitle", Command("textsc", NoEscape(r'Last Course Review:'))))
            ppt.append(NoEscape(r'\begin{itemize}'))
            ppt.append(Command('item', Command("bf", "Last Course Review:")))
            ppt.append(NoEscape(r'\begin{itemize}'))

            nnnn = nlp(show_allsumm_sentence)
            extractor = WINGNUS()
            extractor.load_document(input=nnnn, language='en')
            extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
            extractor.candidate_weighting()
            keywords = [u for u, v in extractor.get_n_best(n=3, stemming=True)]
            result = split_string_by_keywords(show_allsumm_sentence, keywords)
            for tu in result:
                if tu[1] == True:
                    ppt.append(bold(italic(tu[0])))
                else:
                    ppt.append(italic(tu[0]))
        elif result_fl["data"]["src"] == 'zh':
            ppt.append(Command('begin', 'frame'))
            ppt.append(Command("frametitle", Command("textsc", NoEscape(r'回顾上一节课的知识点：'))))
            ppt.append(NoEscape(r'\begin{itemize}'))
            ppt.append(Command('item', Command("bf", "回顾上一节课的知识点：")))
            ppt.append(NoEscape(r'\begin{itemize}'))

            keyphrases = SIFRank(show_allsumm_sentence, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
            keywords = [i[0] for i in keyphrases]
            result = split_string_by_keywords(show_allsumm_sentence, keywords)
            for tu in result:
                if tu[1] == True:
                    ppt.append(bold(italic(tu[0])))
                else:
                    ppt.append(italic(tu[0]))

        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(Command('end', 'frame'))

    if result_fl["data"]["src"] == 'zh':
        for index_title in range(len(titles)):
            titles[index_title] = translate('en', 'zh', titles[index_title])
        with doc.create(Section("接下来我们学习一些重要的知识点，大家注意听讲")):
            doc.append(italic('这篇文章被自动分析，分为多个部分，总结如下。\n'))
        ppt.append(Command('section', '节选部分的摘要'))
        ppt.append(Command('begin', 'frame'))
        ppt.append(Command("frametitle", Command("textsc", NoEscape(r'节选部分摘要的说明'))))
        ppt.append(NoEscape(r'\begin{itemize}'))
        ppt.append(Command('item', Command("bf", "经过分析，这篇本节课程分为{}个部分。".format(len(titles)))))
        ppt.append(NoEscape(r'\begin{itemize}'))
        for title_str in titles:
            ppt.append(Command('item', title_str))
        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(Command('end', 'frame'))
    elif result_fl["data"]["src"] == 'en':
        with doc.create(Section("Important knowledge points and summaries of each excerpt")):
            doc.append(
                italic('The article is automatically analyzed, divided into serval sections, and summarized below.\n'))
        ppt.append(Command('section', 'Section Summary'))
        ppt.append(Command('begin', 'frame'))
        ppt.append(Command("frametitle", Command("textsc", NoEscape(r'Sections Summary Description'))))
        ppt.append(NoEscape(r'\begin{itemize}'))
        ppt.append(Command('item', Command("bf", "After analysis, there are {} sections in this course.".format(
            len(titles)))))
        ppt.append(NoEscape(r'\begin{itemize}'))
        for title_str in titles:
            ppt.append(Command('item', title_str))
        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(NoEscape(r'\end{itemize}'))
        ppt.append(Command('end', 'frame'))

    for index_summ in range(len(summs)):
        if result_fl["data"]["src"] == 'zh':
            summs[index_summ] = translate('en', 'zh', summs[index_summ])

    with doc.create(MiniPage(width=r'0.95\textwidth')):
        for w in range(len(titles)):
            with doc.create(Subsection(str(titles[w]))):
                if result_fl["data"]["src"] == 'zh':
                    #   doc.append(str(summs[w]))
                    doc.append("**同学们，我们现在来学习什么是"+str(titles[w])+"。**\n")

                    keyphrases = SIFRank(str(summs[w]), SIF, zh_model, N=15,
                                         elmo_layers_weight=elmo_layers_weight)
                    keywords = [i[0] for i in keyphrases]
                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            doc.append(bold(tu[0]))
                        else:
                            doc.append(tu[0])

                elif result_fl["data"]["src"] == 'en':
                    doc.append("**Attention students, let's learn what is " + str(titles[w]) + ".**\n")

                    nnnn = nlp(str(summs[w]))
                    extractor = WINGNUS()
                    extractor.load_document(input=nnnn, language='en')
                    extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
                    extractor.candidate_weighting()
                    keywords = [u for u, v in extractor.get_n_best(n=3, stemming=True)]

                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            doc.append(bold(tu[0]))
                        else:
                            doc.append(tu[0])

            if w % 2 == 0:
                ppt.append(Command('begin', 'frame'))
                if result_fl["data"]["src"] == 'zh':
                    ppt.append(Command("frametitle", Command("textsc", NoEscape(r'节选部分的摘要'))))
                    ppt.append(NoEscape(r'\begin{block}{' + str(titles[w]) + '}'))

                    keyphrases = SIFRank(str(summs[w]), SIF, zh_model, N=15,
                                         elmo_layers_weight=elmo_layers_weight)
                    keywords = [i[0] for i in keyphrases]
                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            ppt.append(bold(italic(tu[0])))
                        else:
                            ppt.append(italic(tu[0]))

                    ppt.append(NoEscape(r'\end{block}'))

                elif result_fl["data"]["src"] == 'en':
                    ppt.append(Command("frametitle", Command("textsc", NoEscape(r'Section Summary'))))
                    ppt.append(NoEscape(r'\begin{block}{' + str(titles[w]) + '}'))

                    nnnn = nlp(str(summs[w]))
                    extractor = WINGNUS()
                    extractor.load_document(input=nnnn, language='en')
                    extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
                    extractor.candidate_weighting()
                    keywords = [u for u, v in extractor.get_n_best(n=3, stemming=True)]

                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            ppt.append(bold(tu[0]))
                        else:
                            ppt.append(tu[0])

                    ppt.append(NoEscape(r'\end{block}'))
            else:
                if result_fl["data"]["src"] == 'zh':
                    ppt.append(NoEscape(r'\begin{alertblock}{' + str(titles[w]) + '}'))
                    #   ppt.append(str(summs[w]))
                    keyphrases = SIFRank(str(summs[w]), SIF, zh_model, N=15,
                                         elmo_layers_weight=elmo_layers_weight)
                    keywords = [i[0] for i in keyphrases]
                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            ppt.append(bold(italic(tu[0])))
                        else:
                            ppt.append(italic(tu[0]))

                    ppt.append(NoEscape(r'\end{alertblock}'))
                    ppt.append(Command('end', 'frame'))
                    ppt.append(Command('subsection', str(titles[w - 1]) + ";" + str(titles[w])))
                elif result_fl["data"]["src"] == 'en':
                    ppt.append(NoEscape(r'\begin{alertblock}{' + str(titles[w]) + '}'))
                    ppt.append(str(summs[w]))

                    nnnn = nlp(str(summs[w]))
                    extractor = WINGNUS()
                    extractor.load_document(input=nnnn, language='en')
                    extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
                    extractor.candidate_weighting()
                    keywords = [u for u, v in extractor.get_n_best(n=3, stemming=True)]

                    result = split_string_by_keywords(str(summs[w]), keywords)
                    for tu in result:
                        if tu[1] == True:
                            ppt.append(bold(tu[0]))
                        else:
                            ppt.append(tu[0])

                    ppt.append(NoEscape(r'\end{alertblock}'))
                    ppt.append(Command('end', 'frame'))
                    ppt.append(Command('subsection', str(titles[w - 1]) + ";" + str(titles[w])))
            if w == len(titles) - 1 and w % 2 == 0:
                ppt.append(Command('subsection', str(titles[w])))
                ppt.append(Command('end', 'frame'))


    #======================
    if result_fl["data"]["src"] == 'zh':
        with doc.create(Section("全部课程内容的总结：")):
            doc.append("**最后，我们来回顾一下本节课所学到知识。**\n")
            keyphrases = SIFRank(generated_text_bring_to_last, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
            keywords = [i[0] for i in keyphrases]
            result = split_string_by_keywords(generated_text_bring_to_last, keywords)
            for tu in result:
                if tu[1] == True:
                    doc.append(bold(italic(tu[0])))
                else:
                    doc.append(italic(tu[0]))

    elif result_fl["data"]["src"] == 'en':
        with doc.create(Section("Summary of all course content:")):
            doc.append("**Finally, let's review the knowledge learned in this lesson.**\n")
            nnnn = nlp(generated_text_bring_to_last)
            extractor = WINGNUS()
            extractor.load_document(input=nnnn, language='en')
            extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
            extractor.candidate_weighting()
            keywords = [u for u, v in extractor.get_n_best(n=6, stemming=True)]
            result = split_string_by_keywords(generated_text_bring_to_last, keywords)
            for tu in result:
                if tu[1] == True:
                    doc.append(bold(italic(tu[0])))
                else:
                    doc.append(italic(tu[0]))


    #===================





    if result_fl["data"]['src'] == "zh":
        doc.append(Section("请把你的笔记写在这"))
        doc.generate_pdf(user_ip+'/prezip/note', clean_tex=False)
        ppt.append(Command('begin', 'frame'))
        ppt.append(Command('Huge', Command('centerline', '谢谢观看！')))
    elif result_fl["data"]['src'] == "en":
        doc.append(Section("Write your notes here"))
        doc.generate_pdf(user_ip+'/prezip/note', clean_tex=False)
        ppt.append(Command('begin', 'frame'))
        ppt.append(Command('Huge', Command('centerline', 'Thank you!')))
    ppt.append(Command('end', 'frame'))
    ppt.generate_pdf(clean_tex=False)
    with open(user_ip + "/state.txt", 'w') as f:
        f.write("Completed, you can download these files!")


def your_ajax_view(request):
    # 处理你的逻辑
    result = "算法运行时，请您耐心等待！"
    return HttpResponse(result)

def your_api_url(request):
    data_1 = request.GET.get('key1')
    if not os.path.exists(data_1):
        os.mkdir(data_1)
        file = open(data_1+"/state.txt", "w")
        file.close()
        os.mkdir(data_1+"/prezip")
    with open(data_1+"/state.txt", 'w') as f:
        f.write("算法已经准备就绪")
    return HttpResponse('Successful!')