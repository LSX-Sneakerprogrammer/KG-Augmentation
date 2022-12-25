# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import re
from scipy.sparse import csr_matrix

import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
#torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                # i += 1
                # if i > 100:
                #     break
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class RuleProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_multihop_triples(self, data_dir, data_type, written_dir, K, T, n_fre=10, n_infre=10):
        """See base class."""
        return self._create_multihop_triples(data_dir, data_type, written_dir, K, T, n_fre=n_fre, n_infre=n_infre)


    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_multi_hop_indices(self, tensor1, tensor2):
        tensor_2_hop = tensor1 @ tensor2
        tensor_2_hop_dok = tensor_2_hop.todok()
        tensor_2_hop_coo = tensor_2_hop_dok.tocoo()
        col = list(tensor_2_hop_coo.col.reshape(-1))
        row = list(tensor_2_hop_coo.row.reshape(-1))
        data = tensor_2_hop.data
        # indices = np.argsort(-data)

        return row, col, data

    def save_multi_hop_triples(self, index1, index2, row, col, data, max_num, entities, relations, saved_path):
        count_hop = 0
        if len(row) > max_num:
            indexes = random.sample(range(0, len(row)), max_num)
            with open(saved_path, mode='a', encoding='utf-8') as f:
                for index in indexes:
                    l = str(entities[row[index]]) + '\t' + str(relations[index1]) + '\t' + str(relations[index2]) + '\t' + str(entities[col[index]]) + '\t' + str(data[index]) + '\n'
                    f.write(l)
            count_hop += len(indexes)
        elif len(row) > 0:
            with open(saved_path, mode='a', encoding='utf-8') as f:
                for index in range(len(row)):
                    l = str(entities[row[index]]) + '\t' + str(relations[index1]) + '\t' + str(relations[index2]) + '\t' + str(entities[col[index]]) + '\t' + str(data[index]) + '\n'
                    f.write(l)
            count_hop += len(row)
        return count_hop

    def save_multi_hop_triples_3(self, index1, index2, index3, row, col, data, max_num, entities, relations, saved_path):
        count_hop = 0
        if len(row) > max_num:
            indexes = random.sample(range(0, len(row)), max_num)
            with open(saved_path, mode='w', encoding='utf-8') as f:
                for index in indexes:
                    l = str(entities[row[index]]) + '\t' + str(relations[index1]) + '\t' + str(relations[index2]) + '\t' + str(relations[index3]) + '\t' + str(entities[col[index]]) + '\t' + str(data[index]) + '\n'
                    f.write(l)
            f.close()
            count_hop += len(indexes)
        elif len(row) > 0:
            with open(saved_path, mode='w', encoding='utf-8') as f:
                for index in range(len(row)):
                    l = str(entities[row[index]]) + '\t' + str(relations[index1]) + '\t' + str(relations[index2]) + '\t' + str(relations[index3]) + '\t' + str(entities[col[index]]) + '\t' + str(data[index]) + '\n'
                    f.write(l)
            f.close()
            count_hop += len(row)
        return count_hop

    def _create_multihop_triples(self, data_dir, data_type, written_dir, K, T, n_fre, n_infre):
        relations = self.get_relations(data_dir)
        entities = self.get_entities(data_dir)
        lines = self.get_train_triples(data_dir)
        M = len(relations)
        N = len(entities)

        rel_data = [[] for _ in range(len(relations))]
        for (i, line) in enumerate(lines):
            r_index = relations.index(line[1])
            rel_data[r_index].append(line)

        if data_type == "few-shot":
            logger.info(f"********* Calculate 2-hop triples for few-shot learning************")
            count_hop = 0
            cnt_relations = np.zeros(M)
            for (i, line) in enumerate(lines):
                r_index = relations.index(line[1])
                cnt_relations[r_index] += 1
            sorted_nums = sorted(enumerate(cnt_relations), key=lambda x: x[1], reverse=True)
            idx = [i[0] for i in sorted_nums]
            fre_idx = idx[:n_fre]
            unfre_idx = idx[-n_infre:]
            for index1 in fre_idx:
                for index2 in unfre_idx:
                    logger.info(f"index1 : {index1}, index2 : {index2}")
                    # if index1 == index2:
                    #     continue
                    tensor1_x = []
                    tensor1_y = []
                    tensor2_x = []
                    tensor2_y = []
                    lines1 = rel_data[index1]
                    lines2 = rel_data[index2]
                    max_num = max(len(lines1), len(lines2))
                    max_num = int(max_num / T)
                    for line in lines1:
                        h_index = entities.index(line[0])
                        t_index = entities.index(line[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    tensor1_data = np.ones(len(tensor1_x))
                    tensor1_x = np.array(tensor1_x)
                    tensor1_y = np.array(tensor1_y)
                    tensor1 = csr_matrix((tensor1_data, (tensor1_x, tensor1_y)), shape=(N, N))
                    for line in lines2:
                        h_index = entities.index(line[0])
                        t_index = entities.index(line[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    tensor2_data = np.ones(len(tensor2_x))
                    tensor2_x = np.array(tensor2_x)
                    tensor2_y = np.array(tensor2_y)
                    tensor2 = csr_matrix((tensor2_data, (tensor2_x, tensor2_y)), shape=(N, N))

                    row, col, data = self.get_multi_hop_indices(tensor1, tensor2)
                    inv_row, inv_col, inv_data = self.get_multi_hop_indices(tensor2, tensor1)

                    # saved_path = os.path.join(filename, f'{index1}_{index2}.tsv')
                    # inv_saved_path = os.path.join(filename, f'{index2}_{index1}.tsv')

                    cnt1 = self.save_multi_hop_triples(index1, index2, row, col, data, max_num, entities, relations, written_dir)
                    cnt2 = self.save_multi_hop_triples(index2, index1, inv_row, inv_col, inv_data, max_num, entities, relations, written_dir)
                    count_hop += (cnt1 + cnt2)
                    # count_hop += cnt1

                    logger.info(f"count_hop : {count_hop}, cnt1 : {cnt1}, cnt2 : {cnt2}")


        if data_type == "multihop" and K == 2:
            count_hop = 0
            logger.info(f"********* Calculate 2-hop triples ************")
            for index1 in range(M):
                for index2 in range(M):
                    logger.info(f"index1 : {index1}, index2 : {index2}")
                    if index1 == index2:
                        continue
                    tensor1_x = []
                    tensor1_y = []
                    tensor2_x = []
                    tensor2_y = []
                    lines1 = rel_data[index1]
                    lines2 = rel_data[index2]
                    max_num = max(len(lines1), len(lines2))
                    max_num = int(max_num / T)
                    for line in lines1:
                        h_index = entities.index(line[0])
                        t_index = entities.index(line[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    tensor1_data = np.ones(len(tensor1_x))
                    tensor1_x = np.array(tensor1_x)
                    tensor1_y = np.array(tensor1_y)
                    tensor1 = csr_matrix((tensor1_data, (tensor1_x, tensor1_y)), shape=(N, N))
                    for line in lines2:
                        h_index = entities.index(line[0])
                        t_index = entities.index(line[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    tensor2_data = np.ones(len(tensor2_x))
                    tensor2_x = np.array(tensor2_x)
                    tensor2_y = np.array(tensor2_y)
                    tensor2 = csr_matrix((tensor2_data, (tensor2_x, tensor2_y)), shape=(N, N))

                    row, col, data = self.get_multi_hop_indices(tensor1, tensor2)

                    cnt1 = self.save_multi_hop_triples(index1, index2, row, col, data, max_num, entities, relations, written_dir)
                    count_hop += cnt1

                    logger.info(f"count_hop : {count_hop}, cnt1 : {cnt1}")

        if data_type == "multihop" and K == 3:
            count_hop = 0
            logger.info(f"********* Calculate 3-hop triples ************")
            for index1 in range(M):
                for index2 in range(M):
                    for index3 in range(M):
                        logger.info(f"index1 : {index1}, index2 : {index2}, index3 : {index3}")
                        tensor1_x = []
                        tensor1_y = []
                        tensor2_x = []
                        tensor2_y = []
                        tensor3_x = []
                        tensor3_y = []
                        lines1 = rel_data[index1]
                        lines2 = rel_data[index2]
                        lines3 = rel_data[index3]
                        max_num = max(len(lines1), len(lines2), len(lines3))
                        max_num = int(max_num / T)
                        for line in lines1:
                            h_index = entities.index(line[0])
                            t_index = entities.index(line[2])
                            tensor1_x.append(h_index)
                            tensor1_y.append(t_index)
                            # tensor1[h_index][t_index] += 1
                        tensor1_data = np.ones(len(tensor1_x))
                        tensor1_x = np.array(tensor1_x)
                        tensor1_y = np.array(tensor1_y)
                        tensor1 = csr_matrix((tensor1_data, (tensor1_x, tensor1_y)), shape=(N, N))
                        for line in lines2:
                            h_index = entities.index(line[0])
                            t_index = entities.index(line[2])
                            tensor2_x.append(h_index)
                            tensor2_y.append(t_index)
                            # tensor2[h_index][t_index] += 1
                        tensor2_data = np.ones(len(tensor2_x))
                        tensor2_x = np.array(tensor2_x)
                        tensor2_y = np.array(tensor2_y)
                        tensor2 = csr_matrix((tensor2_data, (tensor2_x, tensor2_y)), shape=(N, N))

                        for line in lines3:
                            h_index = entities.index(line[0])
                            t_index = entities.index(line[2])
                            tensor3_x.append(h_index)
                            tensor3_y.append(t_index)
                            # tensor2[h_index][t_index] += 1
                        tensor3_data = np.ones(len(tensor3_x))
                        tensor3_x = np.array(tensor3_x)
                        tensor3_y = np.array(tensor3_y)
                        tensor3 = csr_matrix((tensor3_data, (tensor3_x, tensor3_y)), shape=(N, N))

                        tensor12 = tensor1 @ tensor2
                        row, col, data = self.get_multi_hop_indices(tensor12, tensor3)

                        cnt1 = self.save_multi_hop_triples_3(index1, index2, index3, row, col, data, max_num, entities, relations, written_dir)
                        count_hop += cnt1

                        logger.info(f"count_hop : {count_hop}, cnt1 : {cnt1}")




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        required=True,
                        # default="./umls",
                        type=str,
                        help="The input data dir. Should contain the rule file.")
    parser.add_argument("--data_type",
                        # required=True,
                        default="multihop",
                        type=str,
                        help="Which dataset needs to be extract")
    parser.add_argument("--K",
                        # required=True,
                        default=2,
                        type=int,
                        help="The number of hops for multi-hop triples")
    parser.add_argument("--n_fre",
                        default=10,
                        type=int,
                        help="The top number of frequent relations")
    parser.add_argument("--n_infre",
                        default=10,
                        type=int,
                        help="The top number of infrequent relations")
    parser.add_argument("--T",
                        default=10,
                        type=int,
                        help="The temperature to control the number of facts")
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    processor = RuleProcessor()
    written_dir = os.path.join(args.data_dir, "multihop_triples.tsv")
    processor.get_multihop_triples(args.data_dir, args.data_type, written_dir, args.K, args.T, args.n_fre, args.n_infre)


              
if __name__ == "__main__":
    main()
