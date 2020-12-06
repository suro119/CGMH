from collections import Counter
from absl import flags  # Added because tf.flags has been depreciated
from absl import app  # Added because tf.flags has been depreciated
import pickle as pkl

FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string('mode','make','[make | use | reverse]')
flags.DEFINE_string('src_path','','')
flags.DEFINE_string('tgt_path','output','')
flags.DEFINE_string('dict_path','dict.pkl','')
flags.DEFINE_integer('dict_size',30000,'')

def main():
    f = open(FLAGS.src_path, 'r')

    # Creates the (word --> index) and (index --> word) dictionary using
    # source file located at 'src_path'
    if FLAGS.mode == 'make':
        words = f.read().split()
        counter = Counter(words)
        common_words = counter.most_common(FLAGS.dict_size)
        common_words = zip(*common_words)[0]  # Extract only the words from list of (word, count) tuples
        word2id = dict(zip(common_words, range(FLAGS.dict_size)))
        id2word = dict(zip(range(FLAGS.dict_size), common_words))

        with open(FLAGS.dict_path, 'wb') as  f:
            pkl.dump([word2id, id2word], f)
        
        in_dict = [(x in word2id) for x in words]
        coverage = sum(in_dict)/len(in_dict)
        print('Coverage: {}'.format(coverage))

    # Converts the entire text source file at 'src_path' into a list of type
    # List[List[int]]. Each inner list is a sentence converted into integer 
    # indices. Saves the result to 'tgt_path'.
    elif FLAGS.mode == 'use':
        Dict, _ = pkl.load(open(FLAGS.dict_path), 'rb')
        UNK = len(Dict)

        # Conversion
        with open(FLAGS.tgt_path, 'wb') as g:
            output_list = []
            for line in f:
                word_list = line.split()
                for i in range(len(word_list)):
                    word = word_list[i]
                    if word in Dict:
                        word_list[i] = Dict[word]
                    else:
                        word_list[i] = UNK  # Unknown token
                output_list.append(word_list)
            # If the output file has the .pkl extension (desired)
            if FLAGS.tgt_path[-3:] == 'pkl':
                pkl.dump(output_list, g)
            # If the output file is a .txt file or or any other type (undesired)
            else:
                for line in output_list:
                    for i in range(len(line)):
                        g.write(str(line[i]))
                        if i < len(line) - 1:
                            g.write(' ')
                        else:
                            g.write('\n')

    # Converts the entire index source file at 'src_path' into a list of type
    # List[List[str]] where each inner list is a list of indices converted into
    # a sentence. Saves the result to 'tgt_path'
    elif FLAGS.mode == 'reverse':
        _, Dict = pkl.load(open(FLAGS.dict_path), 'rb')

        # Conversion
        with open(FLAGS.tgt_path,'wb') as g:
            output_list = []
            for line in f:
                word_list = line.split()
                for i in range(len(word_list)):
                    word = int(word_list[i])
                    if word in Dict:
                        word_list[i] = Dict[word]
                    else:
                        word_list[i] = 'UNK'
                output_list.append(word_list)
            # If the output file has the .pkl extension (desired)
            if FLAGS.tgt_path[-3:] == 'pkl':
                pkl.dump(output_list, g)
            # If the output file is a .txt file or or any other type (undesired)
            else:
                for line in output_list:
                    for i in range(len(line)):
                        g.write(line[i])
                        if i < len(line) - 1:
                            g.write(' ')
                        else:
                            g.write('\n')
    
    f.close()

if __name__ == '__main__':
    app.run(main)