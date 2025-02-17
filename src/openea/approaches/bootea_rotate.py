import gc
import math
import multiprocessing as mp
import random
import time
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from openea.modules.finding.evaluation import early_stop
import openea.modules.train.batch as bat
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.load.kg import KG
from openea.modules.utils.util import load_session
from openea.models.basic_model import BasicModel
from openea.modules.base.initializers import init_embeddings
import openea.modules.load.read as rd
from openea.approaches.bootea import generate_supervised_triples, generate_pos_batch, bootstrapping


class BootEA_RotatE(BasicModel):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None
        self.pi = 3.14159265358979323846
        self.epsilon = 2.0
        self.embedding_range = None

    def init(self):
        self.embedding_range = (self.args.gamma + self.epsilon) / self.args.dim
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2

        # customize parameters
        assert self.args.alignment_module == 'swapping'

        assert self.args.neg_triple_num > 0.0
        assert self.args.truncated_epsilon > 0.0

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.re_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 're_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm, tf.float64)
            self.im_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'im_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm, tf.float64)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm, tf.float64)

    def _generate_scores(self, rh, rr, rt, ih, ir, it, pos=True):
        re_score = rh * rr - ih * ir - rt
        im_score = rh * ir + ih * rr - it
        # print("im_score", im_score.shape)
        scores = tf.stack([re_score, im_score], axis=0)
        # print("scores 1", scores.shape)
        scores = tf.norm(scores, axis=0)
        # print("scores 2", scores.shape)
        scores = tf.reduce_sum(scores, axis=-1)
        # print("scores 3", scores.shape)
        scores = self.args.gamma - scores
        return scores if pos else -scores

    def _generate_loss(self, pos_scores, neg_scores):
        pos_scores = tf.sigmoid(pos_scores)
        neg_scores = tf.sigmoid(neg_scores)
        pos_scores = tf.log(pos_scores)
        neg_scores = tf.log(neg_scores)
        pos_loss = tf.reduce_sum(pos_scores)
        neg_loss = tf.reduce_sum(neg_scores)
        loss = - pos_loss - neg_loss  # / self.args.neg_triple_num
        return loss

    def lookup_all(self, h, r, t):
        re_head = tf.nn.embedding_lookup(self.re_ent_embeds, h)
        re_tail = tf.nn.embedding_lookup(self.re_ent_embeds, t)
        im_head = tf.nn.embedding_lookup(self.im_ent_embeds, h)
        im_tail = tf.nn.embedding_lookup(self.im_ent_embeds, t)
        relation = tf.nn.embedding_lookup(self.rel_embeds, r)
        phase_relation = relation / (self.embedding_range / self.pi)
        re_relation = tf.cos(phase_relation)
        im_relation = tf.sin(phase_relation)
        return re_head, re_relation, re_tail, im_head, im_relation, im_tail

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            prh, prr, prt, pih, pir, pit = self.lookup_all(self.pos_hs, self.pos_rs, self.pos_ts)
            nrh, nrr, nrt, nih, nir, nit = self.lookup_all(self.neg_hs, self.neg_rs, self.neg_ts)

        with tf.name_scope('triple_loss'):
            pos_scores = self._generate_scores(prh, prr, prt, pih, pir, pit, pos=True)
            neg_scores = self._generate_scores(nrh, nrr, nrt, nih, nir, nit, pos=False)
            self.triple_loss = self._generate_loss(pos_scores, neg_scores)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _eval_valid_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.valid_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.test_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def eval_kg1_useful_ent_embeddings(self):
        ent_embeds = self.re_ent_embeds + self.im_ent_embeds
        embeds = tf.nn.embedding_lookup(ent_embeds, self.kgs.useful_entities_list1)
        if self.args.ent_l2_norm:
            embeds = tf.nn.l2_normalize(embeds, 1)
        return embeds.eval(session=self.session)

    def eval_kg2_useful_ent_embeddings(self):
        ent_embeds = self.re_ent_embeds + self.im_ent_embeds
        embeds = tf.nn.embedding_lookup(ent_embeds, self.kgs.useful_entities_list2)
        if self.args.ent_l2_norm:
            embeds = tf.nn.l2_normalize(embeds, 1)
        return embeds.eval(session=self.session)

    def save(self):
        ent_embeds = self.re_ent_embeds.eval(session=self.session) + self.im_ent_embeds.eval(session=self.session)
        ent_embeds = preprocessing.normalize(ent_embeds)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)

    def _define_alignment_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        prh, prr, prt, pih, pir, pit = self.lookup_all(self.new_h, self.new_r, self.new_t)
        pos_scores = self._generate_scores(prh, prr, prt, pih, pir, pit, pos=True)
        pos_scores = tf.sigmoid(pos_scores)
        pos_scores = tf.log(pos_scores)
        pos_loss = tf.reduce_sum(pos_scores)
        self.alignment_loss = - pos_loss
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    def eval_ref_sim_mat(self):
        ent_embeds = self.re_ent_embeds + self.im_ent_embeds
        refs1_embeddings = tf.nn.embedding_lookup(ent_embeds, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(ent_embeds, self.ref_ent2)
        # if self.args.ent_l2_norm:
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1).eval(session=self.session)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1).eval(session=self.session)
        return np.matmul(refs1_embeddings, refs2_embeddings.T)

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
                alignment_feed_dict = {self.new_h: [tr[0] for tr in newly_batch1],
                                       self.new_r: [tr[1] for tr in newly_batch1],
                                       self.new_t: [tr[2] for tr in newly_batch1]}
                alignment_vals = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
                alignment_loss += alignment_vals["loss"]
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue,
                                       neighbors1, neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if (self.early_stop and i >= self.args.min_iter) or i == iter_nums:
                    break
            if i * sub_num >= self.args.start_bp:
                print("bootstrapping")
                labeled_align, entities1, entities2, _ = bootstrapping(self.eval_ref_sim_mat(),
                                                                    self.ref_ent1, self.ref_ent2, labeled_align,
                                                                    self.args.sim_th, self.args.k)
                self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, self.args.align_times)
                if i * sub_num >= self.args.start_valid:
                    self.valid(self.args.stop_metric)
            t1 = time.time()
            if self.args.neg_sampling == "truncated":
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list1,
                                                     neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list2,
                                                     neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
