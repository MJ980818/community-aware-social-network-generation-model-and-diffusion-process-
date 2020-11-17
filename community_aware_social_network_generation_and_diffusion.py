#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install powerlaw


# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import matplotlib.pyplot as plt

# from igraph.clustering import *


# In[2]:


def addNode():
    clusterId = int(np.random.choice(range(1,numberOfClusters+1), size = 1, p = clusterProb))
    global G
    i = G.number_of_nodes()
    G.add_node(i+1, cl=clusterId)
    #print("===addNode===")
    return G


# In[3]:


def addLink(i, j):
    global G
    #print("===addEdge===")
    G.add_edge(int(i), int(j))
    return G


# In[4]:


def randomLink(i):
    global G
    x = int(i)
    y = int(np.random.choice(list(G.nodes), size = 1))
    if(G.has_edge(x, y) | (x==y) ):
        return G
    else:
        #print("===randomLink===")
        return addLink(x,y)


# In[5]:


def randomLinkCl(i):
    global G
    x = int(i)
    target_nodes= []
    
    try:
        for n in list(G.nodes):
            n = int(n)
            [nclid] = G.nodes[n].values()
            [xclid] = G.nodes[x].values()
            if nclid == xclid:
                target_nodes.append(n)
        cG = G.subgraph(target_nodes)
    except KeyError:
        #print("keyerror in rnlkcl")
        return G
    #try:
    y = int(np.random.choice(list(cG.nodes), size = 1))
        #y = cG.nodes[int(np.random.choice(list(cG.nodes), size = 1))]
    #except ValueError:
    #    return G
    if(cG.has_edge(x, y) | (x==y) ):
        return G
    else:
        #print("===randomLinkCl===")
        return addLink(x,y)


# In[6]:


def prefAttach(i):
    global G
    x = int(i)
    sum = 0.0
    tmpprob = [val for (node, val) in G.degree()]
#     tmpprob = list(G.degree())
    for i in range(0, len(tmpprob)):
        tmpprob[i] += 1
        sum += tmpprob[i]
    for i in range(0, len(tmpprob)):
        tmpprob[i] /= sum
    y = int(np.random.choice(list(G.nodes), size = 1, p=tmpprob))
    if(G.has_edge(x, y) | (x==y) ):
        return G
    else:
        #print("===prefAttach===")
        return addLink(x,y)


# In[7]:


def prefAttachCl(i):
    global G
    x = int(i)
    target_nodes= []
    for n in list(G.nodes):
        n = int(n)
        [nclid] = G.nodes[n].values()
        [xclid] = G.nodes[x].values()
        if nclid == xclid:
            target_nodes.append(n)
    cG = G.subgraph(target_nodes)
    sum = 0.0
    tmpprob = [val for (node, val) in cG.degree()]
    for i in range(0, len(tmpprob)):
        tmpprob[i] += 1
        sum += tmpprob[i]
    for i in range(0, len(tmpprob)):
        tmpprob[i] /= sum
    y = int(np.random.choice(list(cG.nodes), size = 1, p=tmpprob))
    if(cG.has_edge(x, y) | (x==y) ):
        return G
    else:
        return addLink(x,y)


# In[8]:


def secNeighbor(i, grph):
    
    sec_neighbor_list = []
    x = i
    try:
        for n in list(grph[x]):
            n = int(n)
            for m in list(grph.nodes):
                m = int(m)
                if(grph.has_edge(n, m)):
                    sec_neighbor_list.append(m)
    except (KeyError, nx.exception.NetworkXError):
        return grph
    return sec_neighbor_list


# In[9]:


def thirdNeighbor(i, grph):
    
    third_neighbor_list = []
    x = i
    try:
        for n in list(grph[x]):
            n = int(n)
            for m in list(grph.nodes):
                m = int(m)
                if(grph.has_edge(n, m)):
                    for p in grph.nodes():
                        p = int(p)
                        if(grph.has_edge(m, p)):
                            third_neighbor_list.append(p)
    except (KeyError, nx.exception.NetworkXError):
            return grph
    return third_neighbor_list


# In[10]:


def close(i, ord):
    global G
    x = int(i)
    ord = int(ord)
    try:
        if(ord==2):
            neighbor_list = secNeighbor(i, G)
        elif(ord==3):
            neighbor_list = thirdNeighbor(i, G)
        y = int(np.random.choice(neighbor_list, size = 1))
    except (KeyError, ValueError):
        return G
    if(G.has_edge(x, y) | (x==y) ):
        return G
    else:
        #print("===closeLink===")
        return addLink(x,y)


# In[11]:


def closeCl(i, ord):
    global G
    x = int(i)
    target_nodes = []
    try:
        for n in list(G.nodes):
            [nclid] = G.nodes[n].values()
            [iclid] = G.nodes[i].values()
            if nclid == iclid:
                target_nodes.append(n)
        cG = G.subgraph(target_nodes)
    except (KeyError, ValueError):
        return G
    try:
        if(ord==2):
            neighbor_list = secNeighbor(i, cG)
        elif(ord==3):
            neighbor_list = thirdNeighbor(i, cG)
        y = int(np.random.choice(neighbor_list, size = 1))
    except (KeyError, ValueError):
        return G
    if(cG.has_edge(x, y) | (x==y) ):
        return G
    else:
        return addLink(x,y)


# In[27]:


def diffProcess(G, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes):
    threshold_value = 0.3
    
    for i in range(1, G.number_of_nodes()+1):
        if(i in step0_nodes):
            G.nodes[i]['th'] = 1.0
        else:
            G.nodes[i]['th'] = 0.0
    
    repeat_cnt = 0
    
    while(repeat_cnt < 3):
    #while(repeat_cnt < 3):
    #while(len(diff_node_list) < G.number_of_nodes()/2):
        repeat_cnt += 1
        print("cnt: ", repeat_cnt)
        for i in range(1, G.number_of_nodes()+1):
            #print('i: ', i)
            sum = 0.0
            neighbor_size = len(list(G[i]))
            if(neighbor_size == 0):
                #print(i,"의 neighbor size : 0")
                continue
            #print('G[i]: ',G[i])
            for n in list(G[i]):
                #if(i == n):
                #    continue
                n = int(n)
                #print('n : ', n, '  G.nodes[n]: ', G.nodes[n], '  G.nodes[n]["th"] :', G.nodes[n]['th'])
                if(G.nodes[n]['th'] == 1):
                    sum += 1.0
                
            sum /= float(neighbor_size)
            if(i not in diff_nodes):
                if(sum >= threshold_value):
                    G.nodes[i]['th'] = 1
                    diff_nodes.append(i)
                    #if(repeat_cnt < 10):
                    if(repeat_cnt == 1):
                        step1_nodes.append(i)
                    #elif(repeat_cnt < 20):
                    elif(repeat_cnt == 2):
                        step2_nodes.append(i)
                    #else:
                    elif(repeat_cnt == 3):
                        step3_nodes.append(i)
                    elif(repeat_cnt == 4):
                        step4_nodes.append(i)
                        
    
    
    for i in range(1, G.number_of_nodes()+1):
        if(i not in diff_nodes):
            not_diff_nodes.append(i)
    
    
    return G


# In[28]:


def degreeDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes):
    degree_sorted = sorted(list(G.nodes()), key = lambda x : len(list(G[x])), reverse=True)
    step0_num = int(G.number_of_nodes() * 0.1)
    for i in range(1, G.number_of_nodes()+1):
        if(len(step0_nodes)==step0_num):
            break
        if(i in degree_sorted[0:step0_num]):
            step0_nodes.append(i)
            diff_nodes.append(i)
    G = diffProcess(G, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes)
    return G


# In[29]:


def randDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes):
    
    print("==========rand diffusion")
    
    cnt = 0
    
    step0_num = int(G.number_of_nodes() * 0.1)
    
    #while(len(step0_nodes)<cluster_size):
    
    while(cnt<step0_num):
    #while(cnt<cluster_size):
        random_node = int(np.random.choice(range(1,G.number_of_nodes()+1), 1))
        if(random_node not in step0_nodes):
            step0_nodes.append(random_node)
            #step0_nodes[cnt] = random_node
            #step0_nodes.append(random_node)
            diff_nodes.append(random_node)
            cnt += 1
            
            
        
    G = diffProcess(G, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes)
    return G
    


# In[30]:


def comDegreeDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes, cl1_sorted, cl2_sorted, cl3_sorted, cl4_sorted, cl5_sorted, cl6_sorted, cl7_sorted, cl8_sorted):
    
    print("==========community Degree diffusion")
    
    """
    global cl1_max_node
    global cl2_max_node
    global cl3_max_node
    global cl4_max_node
    global cl5_max_node
    global cl6_max_node
    global cl7_max_node
    global cl8_max_node
    """
    
    #step0_nodes = [0]*cluster_size
    #max_degree = [0]*cluster_size
    
    #step0_nodes[0] = cl1_max_node
    #step0_nodes[1] = cl2_max_node
    #step0_nodes[2] = cl3_max_node
    
    step0_num = int(G.number_of_nodes() * 0.1)
    each_cl_step0_num = int(step0_num / cluster_size)
    
    if(cluster_size==3):
        for i in range(0, each_cl_step0_num):
            step0_nodes.append(int(cl1_sorted[i][0]))
            step0_nodes.append(int(cl2_sorted[i][0]))
            step0_nodes.append(int(cl3_sorted[i][0]))
            diff_nodes.append(int(cl1_sorted[i][0]))
            diff_nodes.append(int(cl2_sorted[i][0]))
            diff_nodes.append(int(cl3_sorted[i][0]))
            
    elif(cluster_size==8):
        for i in range(0, each_cl_step0_num):
            step0_nodes.append(int(cl1_sorted[i][0]))
            step0_nodes.append(int(cl2_sorted[i][0]))
            step0_nodes.append(int(cl3_sorted[i][0]))
            step0_nodes.append(int(cl4_sorted[i][0]))
            step0_nodes.append(int(cl5_sorted[i][0]))
            step0_nodes.append(int(cl6_sorted[i][0]))
            step0_nodes.append(int(cl7_sorted[i][0]))
            step0_nodes.append(int(cl8_sorted[i][0]))
            diff_nodes.append(int(cl1_sorted[i][0]))
            diff_nodes.append(int(cl2_sorted[i][0]))
            diff_nodes.append(int(cl3_sorted[i][0]))
            diff_nodes.append(int(cl4_sorted[i][0]))
            diff_nodes.append(int(cl5_sorted[i][0]))
            diff_nodes.append(int(cl6_sorted[i][0]))
            diff_nodes.append(int(cl7_sorted[i][0]))
            diff_nodes.append(int(cl8_sorted[i][0]))
            
    
    """
    if(cluster_size==3):
        step0_nodes.append(cl1_max_node)
        step0_nodes.append(cl2_max_node)
        step0_nodes.append(cl3_max_node)
        diff_nodes.append(cl1_max_node)
        diff_nodes.append(cl2_max_node)
        diff_nodes.append(cl3_max_node)
    elif(cluster_size==8):
        step0_nodes.append(cl1_max_node)
        step0_nodes.append(cl2_max_node)
        step0_nodes.append(cl3_max_node)
        step0_nodes.append(cl4_max_node)
        step0_nodes.append(cl5_max_node)
        step0_nodes.append(cl6_max_node)
        step0_nodes.append(cl7_max_node)
        step0_nodes.append(cl8_max_node)
        diff_nodes.append(cl1_max_node)
        diff_nodes.append(cl2_max_node)
        diff_nodes.append(cl3_max_node)
        diff_nodes.append(cl4_max_node)
        diff_nodes.append(cl5_max_node)
        diff_nodes.append(cl6_max_node)
        diff_nodes.append(cl7_max_node)
        diff_nodes.append(cl8_max_node)
    """
   
    G = diffProcess(G, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes)
    return G
    


# In[31]:


def drawNetwork(G, testno, method_num, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, not_diff_nodes):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=step0_nodes, node_color='#F15F5F',node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=step1_nodes, node_color='#FAE555',node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=step2_nodes, node_color='#85E088',node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=step3_nodes, node_color='#76BDDB',node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=step4_nodes, node_color='#D7AA94',node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=not_diff_nodes, node_color='#DAA3DE',node_size=100)
    """
    nx.draw_networkx_nodes(G, pos, nodelist=step0_nodes, node_color="red",node_size=200)
    nx.draw_networkx_nodes(G, pos, nodelist=step1_nodes, node_color="yellow",node_size=200)
    nx.draw_networkx_nodes(G, pos, nodelist=step2_nodes, node_color="green",node_size=200)
    nx.draw_networkx_nodes(G, pos, nodelist=step3_nodes, node_color="skyblue",node_size=200)
    nx.draw_networkx_nodes(G, pos, nodelist=not_diff_nodes, node_color="purple",node_size=200)
    """
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=1.0)

    nx.draw_networkx_labels(G, pos, with_labels=True, font_size=5)
    #nx.draw(G, with_labels=True)
    
    plt.axis("off")
    plt.savefig('fig' + str(testno) + '_' + str(method_num) +'.png', dpi=500)
    plt.show()
    print("step 0 num : ", len(step0_nodes))
    print("step 1 num : ", len(step1_nodes))
    print("step 2 num : ", len(step2_nodes))
    print("step 3 num : ", len(step3_nodes))
    print("step 4 num : ", len(step4_nodes))
    """
    print("step 0: ", step0_nodes, 'num : ', len(step0_nodes))
    print("step 1: ", step1_nodes, 'num : ', len(step1_nodes))
    print("step 2: ", step2_nodes, 'num : ', len(step2_nodes))
    print("step 3: ", step3_nodes, 'num : ', len(step3_nodes))
    print("step 4: ", step4_nodes, 'num : ', len(step4_nodes))
    """
    print("not_diff num : ", len(not_diff_nodes))
    #print("not_diff: ", not_diff_nodes, 'num : ', len(not_diff_nodes))
    
    


# In[32]:


#nOfTests = 240
#nOfTests = 10
#nOfTests = 60
nOfTests = 10
input_data = pd.read_csv('input_for_test4-Copy3.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_for_test4-Copy2.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_for_test4-Copy1.txt', sep='\t', encoding='utf-8')
# input_data = pd.read_csv('input_for_test4.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_for_test3.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_caltech2.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_karate2.txt', sep='\t', encoding='utf-8')
#input_data = pd.read_csv('input_caltech.txt', sep='\t', encoding='utf-8')
# print(input_data)
output = pd.DataFrame(index=range(0,nOfTests), columns=['n','m','cc','pl','modular','diam','alpha'])
diff_output = pd.DataFrame(index=range(0,nOfTests*3), columns=['method','step0','step1','step2','step3','step4','not_diff'])


# In[33]:




G = nx.Graph()
for testno in range(0, nOfTests):
    n = input_data.n[testno]
    m = input_data.m[testno]
    numberOfClusters = input_data.numberOfClusters[testno]
    clusterProb = input_data.clusterProb[testno]
    #global testno
    #print("========clusterProb=======")
    #print(clusterProb)
    newlist = str(clusterProb).split(',')
    #print("========newlist=======")
    #print(newlist)
    
    for index in range(0, len(newlist)):
        newlist[index] = float(newlist[index])
        index += 1

    clusterProb = newlist
    clusterProb = np.array(clusterProb)
    clusterProb /= clusterProb.sum()
    #print("========clusterProb=======")
    #print(clusterProb)
    #print("========revised clusterProb=======")
    #clusterProb = [0.24, 0.5, 0.26]
    #print(clusterProb)
    cluster_size = len(clusterProb)

    rp = input_data.rp[testno]
    pp = input_data.pp[testno]
    c3p = input_data.c3p[testno]
    c4p = input_data.c4p[testno]
    comp = input_data.comp[testno]

    a = list((1-comp) * np.array([rp, pp, c3p, c4p]))
    b = list(comp * np.array([rp, pp, c3p, c4p]))

    pr = a + b
    pr = np.array(pr)
    pr /= pr.sum()
    #print(pr)
    
    freq = 0
    mdn = int(m/n)
    G.clear()
    G = nx.Graph()
    
    link_method = [0]*8

    while(G.number_of_edges() < m):
        freq += 1
        i = np.nan
        p = np.nan

        #print("node num : ", G.number_of_nodes(), "edge num : ", G.number_of_edges())
        
        if( (freq%mdn == 1) & (G.number_of_nodes() < n) ):
            G = addNode()
            i = str(G.number_of_nodes())
            tempP = pr[0:4]
            tempP = np.array(tempP)
            tempP /= tempP.sum()
            tempP = list(tempP)
            p = int(np.random.choice(range(0,4), 1, p = tempP))
        else:
            i = np.random.choice(list(G.nodes()), size = 1)
            i = int(i)
            p = int(np.random.choice(range(0,8), 1, p = pr))
        
        
        if(p == 0):
            randomLink(i)
        elif(p == 4):
            randomLinkCl(i)
        elif(p == 1):
            prefAttach(i)
        elif(p == 5):
            prefAttachCl(i)
        elif(p == 2):
            close(i, 2)
        elif(p == 6):
            closeCl(i, 2)
        elif(p == 3):
            close(i, 3)
        elif(p == 7):
            closeCl(i, 3)
        else:
            print("오류")
    

    
    output.n[testno] = G.number_of_nodes()
    output.m[testno] = G.number_of_edges()
    output.cc[testno] = nx.transitivity(G)
    
    cnt = 0
    avr = 0.0
    diam = 0.0
    for g in nx.connected_component_subgraphs(G):
    #for g in nx.connected_components(G):
        if(len(g)>1):
            cnt += 1
            print("============nx.average_shortest_path_length(g)")
            print(nx.average_shortest_path_length(g))
            avr += nx.average_shortest_path_length(g)
            print("nx.diameter(g)")
            print(nx.diameter(g))
            diam += nx.diameter(g)
        
    avr /= cnt
    diam /= cnt
    output.pl[testno] = avr
    output.diam[testno] = diam
    
    
    cl_1 = []
    cl_2 = []
    cl_3 = []
    cl_4 = []
    cl_5 = []
    cl_6 = []
    cl_7 = []
    cl_8 = []
    
    cl1 = {}
    cl2 = {}
    cl3 = {}
    cl4 = {}
    cl5 = {}
    cl6 = {}
    cl7 = {}
    cl8 = {}
    
    """
    cl1_max_node = 0
    cl2_max_node = 0
    cl3_max_node = 0
    cl4_max_node = 0
    cl5_max_node = 0
    cl6_max_node = 0
    cl7_max_node = 0
    cl8_max_node = 0
    """
    
    
    for n in range(1, G.number_of_nodes()+1):
        [cl],[clId] = G.nodes[n].keys(), G.nodes[n].values()
        node_degree = len(list(G[n]))
        
        if(int(clId) == 1):
            cl1[n] = node_degree
            cl_1.append(n)
        elif(int(clId) == 2):
            cl2[n] = node_degree
            cl_2.append(n)
        elif(int(clId) == 3):
            cl3[n] = node_degree
            cl_3.append(n)
        elif(int(clId) == 4):
            cl4[n] = node_degree
            cl_4.append(n)
        elif(int(clId) == 5):
            cl5[n] = node_degree
            cl_5.append(n)
        elif(int(clId) == 6):
            cl6[n] = node_degree
            cl_6.append(n)
        elif(int(clId) == 7):
            cl7[n] = node_degree
            cl_7.append(n)
        else:
            cl8[n] = node_degree
            cl_8.append(n)
            
    #cl1.items()
    cl1_sorted = sorted(cl1.items(), key = lambda x : x[1], reverse=True)
    cl2_sorted = sorted(cl2.items(), key = lambda x : x[1], reverse=True)
    cl3_sorted = sorted(cl3.items(), key = lambda x : x[1], reverse=True)
    cl4_sorted = sorted(cl4.items(), key = lambda x : x[1], reverse=True)
    cl5_sorted = sorted(cl5.items(), key = lambda x : x[1], reverse=True)
    cl6_sorted = sorted(cl6.items(), key = lambda x : x[1], reverse=True)
    cl7_sorted = sorted(cl7.items(), key = lambda x : x[1], reverse=True)
    cl8_sorted = sorted(cl8.items(), key = lambda x : x[1], reverse=True)
    #[nodeId],[nodeDegree] = G.nodes[n].keys(), G.nodes[n].values()
    
    
    
    """
    for n in range(1, G.number_of_nodes()+1):
        #clId = int(str(G.nodes[n].values()))
        #clId = {str(key): str(value) for key, value in G.nodes[n]}
        [cl],[clId] = G.nodes[n].keys(), G.nodes[n].values()
        if(int(clId) == 1):
            cl1.append(n)
            if(len(list(G[n]))>cl1_max_node):
                cl1_max_node = n
                print('cl1_max_node : ',cl1_max_node)
        elif(int(clId) == 2):
            cl2.append(n)
            if(len(list(G[n]))>cl2_max_node):
                cl2_max_node = n
                print('cl2_max_node : ',cl2_max_node)
        elif(int(clId) == 3):
            cl3.append(n)
            if(len(list(G[n]))>cl3_max_node):
                cl3_max_node = n
                print('cl3_max_node : ',cl3_max_node)
        elif(int(clId) == 4):
            cl4.append(n)
            if(len(list(G[n]))>cl4_max_node):
                cl4_max_node = n
                print('cl4_max_node : ',cl4_max_node)
        elif(int(clId) == 5):
            cl5.append(n)
            if(len(list(G[n]))>cl5_max_node):
                cl5_max_node = n
                print('cl5_max_node : ',cl5_max_node)
        elif(int(clId) == 6):
            cl6.append(n)
            if(len(list(G[n]))>cl6_max_node):
                cl6_max_node = n
                print('cl6_max_node : ',cl6_max_node)
        elif(int(clId) == 7):
            cl7.append(n)
            if(len(list(G[n]))>cl7_max_node):
                cl7_max_node = n
                print('cl7_max_node : ',cl7_max_node)
        else:
            cl8.append(n)
            if(len(list(G[n]))>cl8_max_node):
                cl8_max_node = n
                print('cl8_max_node : ',cl8_max_node)
    
    """
    cllist = []
    cllist.append(cl_1)
    cllist.append(cl_2)
    cllist.append(cl_3)
    cllist.append(cl_4)
    cllist.append(cl_5)
    cllist.append(cl_6)
    cllist.append(cl_7)
    cllist.append(cl_8)
    
    
    cl_modular = nx.algorithms.community.modularity(G, cllist)
    output.modular[testno] = cl_modular
    node_degree = [val for (node, val) in G.degree()]
    fit1 = powerlaw.Fit(node_degree, xmin=5, discrete=False)
    # fit1 = ig.power_law_fit(G.degree(), xmin=5, method='continuous')
    
    output.alpha[testno] = fit1.power_law.alpha
    
    
    print("==========diffusion process")
    
    
    diff_method = 3
    for method_num in range(0, diff_method):
        max_degree = 0
        step0_nodes = []
        step1_nodes = []
        step2_nodes = []
        step3_nodes = []
        step4_nodes = []
    
        diff_nodes= []
        not_diff_nodes = []
        
        if(method_num == 0):
            #method_num += 1
            degree_G = degreeDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes)
            drawNetwork(degree_G, testno, method_num, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, not_diff_nodes)
            
            diff_output.method[testno*3] = method_num
            diff_output.step0[testno*3] = len(step0_nodes)
            diff_output.step1[testno*3] = len(step1_nodes)
            diff_output.step2[testno*3] = len(step2_nodes)
            diff_output.step3[testno*3] = len(step3_nodes)
            diff_output.step4[testno*3] = len(step4_nodes)
            diff_output.not_diff[testno*3] = len(not_diff_nodes)
        elif(method_num == 1):
            #method_num += 1
            step0_num = int(G.number_of_nodes() * 0.1)
            #step0_nodes = [0]*step0_num
            max_degree = [0]*step0_num
            #step0_nodes = [0]*cluster_size
            #max_degree = [0]*cluster_size
            rand_G = randDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes)
            drawNetwork(rand_G, testno, method_num, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, not_diff_nodes)
            
            diff_output.method[testno*3+1] = method_num
            diff_output.step0[testno*3+1] = len(step0_nodes)
            diff_output.step1[testno*3+1] = len(step1_nodes)
            diff_output.step2[testno*3+1] = len(step2_nodes)
            diff_output.step3[testno*3+1] = len(step3_nodes)
            diff_output.step4[testno*3+1] = len(step4_nodes)
            diff_output.not_diff[testno*3+1] = len(not_diff_nodes)
        else:
            #method_num += 1
            comDegree_G = comDegreeDiff(G, max_degree, cluster_size, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, diff_nodes, not_diff_nodes, cl1_sorted, cl2_sorted, cl3_sorted, cl4_sorted, cl5_sorted, cl6_sorted, cl7_sorted, cl8_sorted)
            drawNetwork(comDegree_G, testno, method_num, step0_nodes, step1_nodes, step2_nodes, step3_nodes, step4_nodes, not_diff_nodes)
            
            diff_output.method[testno*3+2] = method_num
            diff_output.step0[testno*3+2] = len(step0_nodes)
            diff_output.step1[testno*3+2] = len(step1_nodes)
            diff_output.step2[testno*3+2] = len(step2_nodes)
            diff_output.step3[testno*3+2] = len(step3_nodes)
            diff_output.step4[testno*3+2] = len(step4_nodes)
            diff_output.not_diff[testno*3+2] = len(not_diff_nodes)
    print(diff_output)
    
    print(output)
    
    testno += 1
#output.to_csv("output_copy1.csv", header=True)
output.to_csv("output_copy2.csv", header=True)
diff_output.to_csv("diff_output_4_2.csv", header=True)
#output.to_csv("output.txt", header=True)
#diff_output.to_csv("diff_output.txt", header=True)


# In[59]:





# In[ ]:




