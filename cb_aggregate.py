import time
import os
import sys
import numpy as np
import tensorflow as tf
import h5py


MEM_DIR = './results'


np.random.seed(0)
keys = sys.argv[1]
metric = sys.argv[2]
num_images = int(sys.argv[3])
adversarials = sys.argv[4]
while True:
    for adv in adversarials.split(','):
        print(adv)
        adversarial = (adv == 'True')
        for key in keys.split(','):
            bound = 'Bernstein'
            truncation = 0.2
            if metric == 'logit':
                truncation = 3443
            max_sample_size = 128
            ## Experiment Directory
            experiment_dir = os.path.join(
                MEM_DIR, 
                'NShap/inceptionv3/{}_{}_new'.format(metric, key))
            if not tf.gfile.Exists(experiment_dir):
                tf.gfile.MakeDirs(experiment_dir)
            if max_sample_size is None or max_sample_size > num_images:
                max_sample_size = num_images
            experiment_name = 'cb_{}_{}_{}'.format(bound, truncation, max_sample_size)
            if adversarial:
                experiment_name = 'ADV' + experiment_name
            cb_dir = os.path.join(experiment_dir, experiment_name)
            if not tf.gfile.Exists(cb_dir):
                tf.gfile.MakeDirs(cb_dir)
            ##
            if metric == 'accuracy':
                R = 1.
            elif metric == 'xe_loss':
                R = np.log(1000)
            elif metric == 'binary':
                R = 1.
            elif metric == 'logit':
                R = 10.
            else:
                raise ValueError('Invalid metric!')
            top_k = 100
            delta = 0.2
            ## Start
            if not tf.gfile.Exists(os.path.join(experiment_dir, 'players.txt')):
                print('Does not exist!')
                continue
            players = open(os.path.join(
                experiment_dir, 'players.txt')).read().split(',')
            players = np.array(players)
            if not tf.gfile.Exists(os.path.join(cb_dir, 'chosen_players.txt')):
                open(os.path.join(cb_dir, 'chosen_players.txt'), 'w').write(','.join(
                    np.arange(len(players)).astype(str)))

            results = np.sort([saved for saved in tf.gfile.ListDirectory(cb_dir)
                               if 'agg' not in saved and '.h5' in saved])
            squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
            max_vals, min_vals = -np.ones(len(players)), np.ones(len(players))
            for result in results:
                try:
                    with h5py.File(os.path.join(cb_dir, result), 'r') as foo:
                        mem_tmc = foo['mem_tmc'][:]
                except:
                    continue
                if not len(mem_tmc):
                    continue
                sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
                squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
                counts += np.sum(mem_tmc != -1, 0)
                #temp = mem_tmc * (mem_tmc != -1) - 1000 * (mem_tmc == -1)
                #max_vals = np.maximum(max_vals, np.max(temp, 0))
                #temp = mem_tmc * (mem_tmc != -1) + 1000 * (mem_tmc == -1)
                #min_vals = np.minimum(min_vals, np.min(temp, 0))
            counts = np.clip(counts, 1e-12, None)
            vals = sums / (counts + 1e-12)
            variances = R * np.ones_like(vals)
            variances[counts > 1] = squares[counts > 1]
            variances[counts > 1] -= (sums[counts > 1] ** 2) / counts[counts > 1]
            variances[counts > 1] /= (counts[counts > 1] - 1)
            if np.max(counts) == 0:
                os.remove(os.path.join(cb_dir, result))
            cbs = R * np.ones_like(vals)
            if bound == 'Hoeffding':
                cbs[counts > 1] = R * np.sqrt(np.log(2 / delta) / (2 * counts[counts > 1]))
            elif bound == 'Bernstein':
                # From: http://arxiv.org/pdf/0907.3740.pdf
                cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / delta) / counts[counts > 1]) +\
                7/3 * R * np.log(2 / delta) / (counts[counts > 1] - 1)
            thresh = (vals)[np.argsort(vals)[-top_k - 1]]
            chosen_players = np.where(
                ((vals - cbs) < thresh) * ((vals + cbs) > thresh))[0]
            print(cb_dir, np.mean(counts), len(chosen_players))
            open(os.path.join(cb_dir, 'chosen_players.txt'), 'w').write(
                ','.join(chosen_players.astype(str)))
            open(os.path.join(cb_dir, 'variances.txt'), 'w').write(
                ','.join(variances.astype(str)))
            open(os.path.join(cb_dir, 'vals.txt'), 'w').write(
                ','.join(vals.astype(str)))
            open(os.path.join(cb_dir, 'counts.txt'), 'w').write(
                ','.join(counts.astype(str)))
            if len(chosen_players) == 1:
                break
