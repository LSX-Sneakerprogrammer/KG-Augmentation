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
    
    def get_implicit_triples(self, rule_dir, data_dir, data_type, written_dir, n_body, cs, T):
        """See base class."""
        return self._create_implicit_triples(
            self._read_tsv(rule_dir), data_type, data_dir, written_dir, n_body, cs, T)


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

    def _create_implicit_triples(self, lines, data_type, data_dir, written_dir, n_body, cs, T):
        cnt = 0
        pattern1 = re.compile('\([^()]+\)')
        if data_type == "wn18rr":
            pattern2 = re.compile('\_[^()]+.(?=\()')
        elif data_type == "umls" or data_type == "nell":
            pattern2 = re.compile('[a-z][^()]+.(?=\()')
        elif data_type == "fb15k":
            pattern2 = re.compile('\/[^()]+.(?=\()')
        else:
            raise ValueError("The format of the data type is not correct.")

        relations = self.get_relations(data_dir)
        entities = self.get_entities(data_dir)
        training_lines = self.get_train_triples(data_dir)

        rel_data = [[] for _ in range(len(relations))]
        logger.info(f"********* Calculate Multi-hop triples ************")
        for (i, line) in enumerate(training_lines):
            r_index = relations.index(line[1])
            rel_data[r_index].append(line)

        for (i, line) in enumerate(lines):
            support_body = int(line[0])
            support_head = int(line[1])
            confidence = float(line[2])
            rules = line[3]
            num_head = int(support_head / T)
            num_not_head = int(support_body / T) - num_head
            # rules = "_also_see(X,Y) <= _also_see(A,X), _also_see(B,A), _also_see(Y,B)"

            if support_body < n_body:
                continue

            if confidence < cs:
                continue

            head, body = rules.split("<=")
            head = head.strip()
            body = body.strip()

            body_entities = re.findall(pattern1, body)
            head_entities = re.findall(pattern1, head)

            if len(body_entities) <= 1:
                continue

            filter_flag = False
            if len(head_entities[0]) > 5:
                continue

            for item in body_entities:
                if len(item) > 5:
                    filter_flag = True
                    break

            if filter_flag:
                continue

            body_relations = re.findall(pattern2, body)
            head_relations = re.findall(pattern2, head)

            body_indexs = [relations.index(relation) for relation in body_relations]
            body_pairs = [[pair[1], pair[3]] for pair in body_entities]

            N = len(body_entities)
            M = len(entities)

            if N == 2:
                r_body1 = body_indexs[0]
                r_body2 = body_indexs[1]
                r_head = relations.index(head_relations[0])
                tensor3 = np.zeros((M, M))
                tensor1_x = []
                tensor1_y = []
                tensor2_x = []
                tensor2_y = []
                tensor1_data = np.ones(len(rel_data[r_body1]))
                tensor2_data = np.ones(len(rel_data[r_body2]))
                for l in rel_data[r_head]:
                    h_index = entities.index(l[0])
                    t_index = entities.index(l[2])
                    tensor3[h_index][t_index] += 1
                if body_pairs[0][0] == 'X' and body_pairs[1][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                elif body_pairs[0][0] == 'X' and body_pairs[1][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                else:
                    raise ValueError("The format of the rule is not right.")
                tensor1_x = np.array(tensor1_x)
                tensor1_y = np.array(tensor1_y)
                tensor1 = csr_matrix((tensor1_data, (tensor1_x, tensor1_y)), shape=(M, M))
                tensor2_x = np.array(tensor2_x)
                tensor2_y = np.array(tensor2_y)
                tensor2 = csr_matrix((tensor2_data, (tensor2_x, tensor2_y)), shape=(M, M))
                row, col, data = self.get_multi_hop_indices(tensor1, tensor2)

                head_list = []
                not_head_list = []

                for k in range(len(row)):
                    if tensor3[row[k]][col[k]] > 0:
                        head_list.append(k)
                    else:
                        not_head_list.append(k)

                cnt_head = 0
                cnt_body = 0

                with open(written_dir, mode='a', encoding='utf-8') as f:
                    if len(head_list) > num_head:
                        indexes = random.sample(head_list, num_head)
                        for index in indexes:
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_head += 1
                            cnt_body += 1
                            f.write(l)
                    else:
                        for index in range(len(head_list)):
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_head += 1
                            cnt_body += 1
                            f.write(l)

                    if len(not_head_list) > num_not_head:
                        indexes = random.sample(not_head_list, num_not_head)
                        for index in indexes:
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_body += 1
                            f.write(l)
                    else:
                        for index in range(len(head_list)):
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_body += 1
                            f.write(l)
                cnt += cnt_body
                logger.info(
                    f"Head : {support_head}, Body : {support_body}, Extract Head Triples : {cnt_head}, extract total Body Triples : {cnt_body}, total triples : {cnt}")

            if N == 3:
                M = len(entities)
                tensor4 = np.zeros((M, M))
                tensor1_x = []
                tensor1_y = []
                tensor2_x = []
                tensor2_y = []
                tensor3_x = []
                tensor3_y = []
                r_body1 = body_indexs[0]
                r_body2 = body_indexs[1]
                r_body3 = body_indexs[2]
                r_head = relations.index(head_relations[0])
                tensor1_data = np.ones(len(rel_data[r_body1]))
                tensor2_data = np.ones(len(rel_data[r_body2]))
                tensor3_data = np.ones(len(rel_data[r_body3]))
                for l in rel_data[r_head]:
                    h_index = entities.index(l[0])
                    t_index = entities.index(l[2])
                    tensor4[h_index][t_index] += 1
                if body_pairs[0][0] == 'X' and body_pairs[1][0] == 'A' and body_pairs[2][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(h_index)
                        tensor3_y.append(t_index)
                        # tensor3[h_index][t_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][0] == 'A' and body_pairs[2][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(h_index)
                        tensor3_y.append(t_index)
                        # tensor3[h_index][t_index] += 1
                elif body_pairs[0][0] == 'X' and body_pairs[1][1] == 'A' and body_pairs[2][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(h_index)
                        tensor3_y.append(t_index)
                        # tensor3[h_index][t_index] += 1
                elif body_pairs[0][0] == 'X' and body_pairs[1][0] == 'A' and body_pairs[2][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(t_index)
                        tensor3_y.append(h_index)
                        # tensor3[t_index][h_index] += 1
                elif body_pairs[0][0] == 'X' and body_pairs[1][1] == 'A' and body_pairs[2][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(h_index)
                        tensor1_y.append(t_index)
                        # tensor1[h_index][t_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(t_index)
                        tensor3_y.append(h_index)
                        # tensor3[t_index][h_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][0] == 'A' and body_pairs[2][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(h_index)
                        tensor2_y.append(t_index)
                        # tensor2[h_index][t_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(t_index)
                        tensor3_y.append(h_index)
                        # tensor3[t_index][h_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][1] == 'A' and body_pairs[2][0] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(t_index)
                        tensor3_y.append(h_index)
                        # tensor3[t_index][h_index] += 1
                elif body_pairs[0][1] == 'X' and body_pairs[1][1] == 'A' and body_pairs[2][1] == 'Y':
                    for l in rel_data[r_body1]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor1_x.append(t_index)
                        tensor1_y.append(h_index)
                        # tensor1[t_index][h_index] += 1
                    for l in rel_data[r_body2]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor2_x.append(t_index)
                        tensor2_y.append(h_index)
                        # tensor2[t_index][h_index] += 1
                    for l in rel_data[r_body3]:
                        h_index = entities.index(l[0])
                        t_index = entities.index(l[2])
                        tensor3_x.append(h_index)
                        tensor3_y.append(t_index)
                        # tensor3[h_index][t_index] += 1
                else:
                    raise ValueError("The format of the rule is not right.")

                tensor1_x = np.array(tensor1_x)
                tensor1_y = np.array(tensor1_y)
                tensor1 = csr_matrix((tensor1_data, (tensor1_x, tensor1_y)), shape=(M, M))
                tensor2_x = np.array(tensor2_x)
                tensor2_y = np.array(tensor2_y)
                tensor2 = csr_matrix((tensor2_data, (tensor2_x, tensor2_y)), shape=(M, M))
                tensor3_x = np.array(tensor3_x)
                tensor3_y = np.array(tensor3_y)
                tensor3 = csr_matrix((tensor3_data, (tensor3_x, tensor3_y)), shape=(M, M))
                tensor12 = tensor1 @ tensor2
                row, col, data = self.get_multi_hop_indices(tensor12, tensor3)

                head_list = []
                not_head_list = []

                for k in range(len(row)):
                    if tensor4[row[k]][col[k]] > 0:
                        head_list.append(k)
                    else:
                        not_head_list.append(k)

                # print(f"head : {len(head_list)}, not head : {len(not_head_list)}")

                cnt_head = 0
                cnt_body = 0

                with open(written_dir, mode='a', encoding='utf-8') as f:
                    if len(head_list) > num_head:
                        indexes = random.sample(head_list, num_head)
                        for index in indexes:
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_head += 1
                            cnt_body += 1
                            f.write(l)
                    else:
                        for index in range(len(head_list)):
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_head += 1
                            cnt_body += 1
                            f.write(l)

                    if len(not_head_list) > num_not_head:
                        indexes = random.sample(not_head_list, num_not_head)
                        for index in indexes:
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_body += 1
                            f.write(l)
                    else:
                        for index in range(len(head_list)):
                            l = str(entities[row[index]]) + '\t' + str(head_relations[0]) + '\t' + str(
                                entities[col[index]]) + '\n'
                            cnt_body += 1
                            f.write(l)
                cnt += cnt_body
                logger.info(
                    f"Head : {support_head}, Body : {support_body}, Extract Head Triples : {cnt_head}, extract total Body Triples : {cnt_body}, total triples : {cnt}")




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        required=True,
                        # default="./rule_umls",
                        type=str,
                        help="The input data dir. Should contain the rule file.")
    parser.add_argument("--data_type",
                        required=True,
                        # default="umls",
                        type=str,
                        help="Which dataset needs to be extract")
    parser.add_argument("--cs",
                        required=True,
                        # default=0.85,
                        type=float,
                        help="The minimum confidence score for filtering rules")
    parser.add_argument("--n_body",
                        default=100,
                        type=int,
                        help="The minimum number of body for filtering rules")
    parser.add_argument("--T",
                        default=1,
                        type=int,
                        help="The temperature to control the number of facts")
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)



    processor = RuleProcessor()

    rule_dir = os.path.join(args.data_dir, "rules.tsv")
    written_dir = os.path.join(args.data_dir, "implicit_triples.tsv")
    processor.get_implicit_triples(rule_dir, args.data_dir, args.data_type, written_dir, args.n_body, args.cs, args.T)


if __name__ == "__main__":
    main()
