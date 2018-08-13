import time
import sys
import os
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics


from rnn_model import ModelConfig, DSSMModel
from data.loader_data import batch_iter, build_vocab, read_category, read_vocab, process_file

base_dir = 'data'
vocab_dir = os.path.join(base_dir, 'tesla_vocab.txt')
train_title_dir = os.path.join(base_dir, 'tesla_title.txt')
train_word_dir = os.path.join(base_dir, 'tesla_word.txt')
train_label_dir = os.path.join(base_dir, 'tesla_label.txt')

val_title_dir = os.path.join(base_dir, 'tesla_val_title.txt')
val_word_dir = os.path.join(base_dir, 'tesla_val_word.txt')
val_label_dir = os.path.join(base_dir, 'tesla_val_label.txt')

test_title_dir = os.path.join(base_dir, 'tesla_test_title.txt')
test_word_dir = os.path.join(base_dir, 'tesla_test_word.txt')
test_label_dir = os.path.join(base_dir, 'tesla_test_label.txt')

save_dir = 'checkpoints/dssm'
save_path = os.path.join(save_dir, 'model_result')

def get_time_dif(start_time):
    """:param get time """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(title_batch, word_batch, label_batch, keep_prob):
    feed_dict = {
        model.input_title: title_batch,
        model.input_word: word_batch,
        model.input_label: label_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, title_, word_, label_):
    data_len = len(title_)
    batch_eval = batch_iter(title_, word_, label_, 63)
    total_loss = 0.0
    total_acc = 0.0
    for title_batch, word_batch, label_batch in batch_eval:
        batch_len = len(title_batch)
        feed_dict = feed_data(title_batch,word_batch,label_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc  += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard, 重新训练时注意删除tensorboard的文件夹，防止图覆盖
    tensorboard_dir = 'tensorboard/model-DSSM'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("loading training data...")

    start_time = time.time()
    title_train, word_train, label_train = process_file(train_title_dir, train_word_dir, train_label_dir, word_to_id, cat_to_id)
    title_val, word_val, label_val = process_file(val_title_dir, val_word_dir, val_label_dir, word_to_id, cat_to_id)
    time_dif = get_time_dif(start_time)
    print("Time usage:",time_dif)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000 # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(title_train, word_train, label_train, config.batch_size)
        for title_batch, word_batch, label_batch in batch_train:
            feed_dict = feed_data(title_batch, word_batch, label_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0 :
                # 每多少轮次将结果写入 tensorboard sclar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0 :
                # print 当前模型性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, title_val, word_val, label_val)
            #
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improve_str = '*'
                else:
                    improve_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                    'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improve_str))

            session.run(model.optim, feed_dict=feed_dict) # run model
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期得不到提升， 提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    title_test, word_test, label_test = process_file(test_title_dir, test_word_dir, test_label_dir, word_to_id, cat_to_id)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    print('Testing...')
    loss_test, acc_test = evaluate(session, title_test, word_test, label_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(title_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    label_test_cls = np.argmax(label_test, 1)
    label_pred_cls = np.zeros(shape=data_len, dtype=np.int32)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_title: title_test[start_id:end_id],
            model.input_word: word_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        label_pred_cls[start_id:end_id] = session.run(model.label_pred, feed_dict=feed_dict)

        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(label_test_cls, label_test_cls, target_names=categories))

        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring DSSM model...')
    config = ModelConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_title_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    # print(config.vocab_size)
    model = DSSMModel(config)
    train()
    if sys.argv[1] == 'train':
        train()
    else:
        test()

