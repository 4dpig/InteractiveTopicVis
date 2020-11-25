import re

pattern = re.compile(r'iteration: (\d+) of max_iter: 1000, perplexity: (\d+\.\d+)')
iter_list = []
error_list = []
with open("lda_data.txt", 'r') as lda_data:
    for line in lda_data.readlines():
        match_result = pattern.match(line)
        match1 = match_result.group(1)
        iter_list.append(int(match1))
        match2 = match_result.group(2)
        error_list.append(float(match2))

with open("lda_data.txt", 'w') as lda_data:
    lda_data.write(str(iter_list))
    lda_data.write(str(error_list))


