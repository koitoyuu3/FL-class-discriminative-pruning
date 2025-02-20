import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import L1FilterPruner, L1FilterPrunerMasker
from nni.algorithms.compression.pytorch.pruning.structured_pruning_masker import L1FilterPrunerMasker
from nni.algorithms.compression.pytorch.pruning.one_shot_pruner import L1FilterPruner


# %%

def acculumate_feature(model, loader, stop: int):
    model = model.cuda()
    features = {}
    classes = []

    def hook_func(m, x, y, name, feature_iit):
        # print(name, y.shape) # ([256, 64, 8, 8])
        '''ReLU'''
        f = F.relu(y)
        # f = y
        '''Average Pool'''
        feature = F.avg_pool2d(f, f.size()[3])
        # print(feature.shape) # ([256, 64, 1, 1])
        feature = feature.view(f.size()[0], -1)
        # print(feature.shape) # ([256, 64])
        feature = feature.transpose(0, 1)
        # print(feature.shape)
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)

    hook = functools.partial(hook_func, feature_iit=features)

    handler_list = []
    # 遍历模型的所有模块，对于每一个卷积层，注册一个前向钩子，用于在模型前向传播时提取特征，并将这些钩子的句柄存储在handler_list列表中。
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # if not isinstance(m, nn.Linear):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)

    # 在此循环中，对数据加载器进行迭代，获取每个批次的输入数据和目标标签，并通过模型进行前向传播以提取特征。当达到指定的stop批次时，停止迭代。
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        # if batch_idx % (10) == 0:
        print('batch_idx', batch_idx)
        model.eval()
        classes.extend(targets.numpy())
        with torch.no_grad():
            model(inputs.cuda())
    [k.remove() for k in handler_list]
    '''Image-wise Activation'''
    return features, classes


# TODO
def calculate_cp(features: dict, classes: list, dataset: str, coe: int, unlearn_class: int):
    # print(len(classes))
    features_class_wise = {}
    tf_idf_map = {}
    if dataset == 'cifar10':
        class_num = 10
    if dataset == 'cifar100':
        class_num = 100
    list_classes_address = []
    for z in range(class_num):
        address_index = [x for x in range(len(classes)) if classes[x] == z]
        list_classes_address.append([z, address_index])
    dict_address = dict(list_classes_address)
    for fea in features:
        '''Class-wise Activation'''
        class_wise_features = torch.zeros(class_num, features[fea].shape[0])
        image_wise_features = features[fea].transpose(0, 1)
        for i, v in dict_address.items():
            for j in v:
                class_wise_features[i] += image_wise_features[j]
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)
        features_class_wise[fea] = class_wise_features.transpose(0, 1)
        # print(features_class_wise[fea].shape) # ([64, 10])
        '''TF-IDF'''
        calc_tf_idf(features_class_wise[fea], fea, coe=coe,
                    unlearn_class=unlearn_class, tf_idf_map=tf_idf_map)
        # print(tf_idf_map[fea].shape)
    return tf_idf_map


# c - filters; n - classes
# feature = [c, n] ([64, 10])
def calc_tf_idf(feature, name: str, coe: int, unlearn_class: int, tf_idf_map: dict):
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0)
    # print(feature_sum.shape) # ([10])
    balance_coe = np.log((feature.shape[0] / coe) * np.e) if coe else 1.0
    # print(feature.shape, name, coe)
    tf = (feature / sum_on_filters) * balance_coe
    # print(tf.shape) # ([64, 10])
    tf_unlearn_class = tf.transpose(0, 1)[unlearn_class]
    # print(tf_unlearn_class.shape)

    # calc idf for filters
    classes_quant = float(feature.shape[1])
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1)
    # print(mean_on_classes.shape) # ([64, 1])
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor)
    # print(inverse_on_classes.shape) # ([64])
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0))
    # print(idf.shape) # ([64])

    importance = tf_unlearn_class * idf
    # print(importance.shape) # ([64])
    tf_idf_map[name] = importance


def get_threshold_by_sparsity(mapper: dict, sparsity: float):
    assert 0 < sparsity < 1
    # print(len(mapper.values())) # 19
    tf_idf_array = torch.cat([v for v in mapper.values()], 0)
    # print(tf_idf_array.shape) # ([688])
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0] * (1 - sparsity)))[0].min()
    return threshold


class TFIDFPruner(L1FilterPruner):
    def __init__(self, model, config_list, cp_config: dict, pruning_algorithm='l1',
                 optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = TFIDFMasker(model, self, threshold=cp_config["threshold"],
                                  tf_idf_map=cp_config["map"], **algo_kwargs)

    def update_masker(self, model, threshold, mapper):
        self.masker = TFIDFMasker(model, self, threshold=threshold, tf_idf_map=mapper)


class TFIDFMasker(L1FilterPrunerMasker):
    def __init__(self, model, pruner, threshold, tf_idf_map, preserve_round=1, dependency_aware=False):
        super().__init__(model, pruner, preserve_round, dependency_aware)
        self.threshold = threshold
        self.tf_idf_map = tf_idf_map

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter -> importance for each filter
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)

        mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
                      :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        if wrapper.name.split('.')[-1] == 'module':
            name = wrapper.name[0:-7]
        # print(name)
        w_tf_idf_structured = self.tf_idf_map[name]
        return w_tf_idf_structured

