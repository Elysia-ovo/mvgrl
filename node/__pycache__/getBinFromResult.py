import bz2
import pickle
import gzip
import io
import os
import scipy.sparse as scisp
import numpy as np

# 从一开始的标签进行分箱的获得的脚本
def gen_bins(fastafile, resultfile, outputdir):
    # read fasta file
    sequences = {}
    if fastafile.endswith("gz"):
        with gzip.open(fastafile, 'r') as f:
            for line in f:
                line = str(line, encoding="utf-8")
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(' ', 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    else:
        with open(fastafile, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(' ', 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    dic = {}
    with open(resultfile, "r") as f:
        for line in f:
            contig_name, cluster_name = line.strip().split('\t')
            try:
                dic[cluster_name].append(contig_name)
            except:
                dic[cluster_name] = []
                dic[cluster_name].append(contig_name)
    print("Writing bins in \t{}".format(outputdir))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    bin_name = 0
    for _, cluster in dic.items():
        if bin_name < 10:
            bin = 'BIN' + '000' + str(bin_name) + '.fa'
        elif bin_name >= 10 and bin_name < 100:
            bin = 'BIN' + '00' + str(bin_name) + '.fa'
        elif bin_name >= 100 and bin_name < 1000:
            bin = 'BIN' + '0' + str(bin_name) + '.fa'
        else:
            bin = 'BIN'+str(bin_name) + '.fa'
        binfile = os.path.join(outputdir, "{}".format(bin))
        with open(binfile, "w") as f:
            for contig_name in cluster:
                contig_name = ">"+contig_name
                try:
                    sequence = sequences[contig_name]
                except:
                    continue
                f.write(contig_name+"\n")
                f.write(sequence+"\n")
        bin_name += 1
def get_result_bin(result_bin_path):
    label = []
    # 聚类得到的标签
    with open(result_bin_path, 'r') as f:
        for line in f:
            label.append(int(line.strip()))
    print(f"Minimum label value: {min(label)}")
    f.close()
    print(label)
    print(len(label))
    import csv
    contigName = []
    with open('/home/zhaozhimiao/ldd/MVGC/GetBin/4ker_title.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            contigName.append(row[1].split()[0][1:])
    csvfile.close()
    # print(contigName)
    # print(len(contigName))
    # group_label = []
    # for i in label:
    #     group_label.append('group'+str(i))

    # print(group_label)
    with open('/home/zhaozhimiao/ldd/MVGC/Result/cluster.txt', 'w') as out:
        for i in range(len(label)):
            out.write(contigName[i] + '\t' + 'group'+str(label[i]-1))
            out.write('\n')

    gen_bins('/home/zhaozhimiao/ldd/MVGC/GetBin/contigs_over_1000.fasta',
             '/home/zhaozhimiao/ldd/MVGC/Result/cluster.txt', '/home/zhaozhimiao/ldd/MVGC/Result')
    
get_result_bin('/home/zhaozhimiao/ldd/MVGC_result/labels_fin_epoch_400.txt')