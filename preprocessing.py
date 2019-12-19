# recursive function that goes through each json file
def fill():
        global edge_list
        global tensor_list
        # current location in json string
        global location
        # total nodes found
        global total
        # root node
        global input_1          
        
        # grab root token
        item = input_1[location]
        # get int value of token
        index = name_to_int["'"+item+"'"]
        # get tensor for the value
        tensor_to_add = vector_tensor.data[index]
        g.add_nodes(1)

        tensor_list.append(tensor_to_add)
        # where current node is stored
        current_node = total
        total += 1
        location += 1
        # go to next index
        item = input_1[location]

        # if child
        if item == '{':
            # keep moving
            location += 1
            # connect the child to the parent
            edge_list.append((total, current_node))
            # insert child into graph
            fill()
            # get next term
            item = input_1[location]
            # if second child
            if item == '{':
                location += 1
                edge_list.append((total, current_node))
                fill()
                item = input_1[location]
        # if closing
        if item == '}':
            location += 1
            return
        
data = pd.read_json('preprocessed_progs_test.json')

# vocabulary
i = set()

# go through target tree and grab each unique token
for item in data["target_ast"]:
    # get rid of anything that is not a token or root
    x = str(item).replace("[","").replace("]","").replace("{","").replace("}","").replace("''","").replace(":","").replace(",","").split()
    for num,h in enumerate(x, start = 0):
        if(h == "'root'"):
            i.add(x[num + 1])

# go through source tree and grab each unique token
for item in data["source_ast"]:
    x = str(item).replace("[","").replace("\\n","").replace("]","").replace("{","").replace("}","").replace("''","").replace(":","").replace(",","").split()
    for num,h in enumerate(x, start = 0):
        if(h == "'root'"):
            i.add(x[num + 1])

# map each token to an int
name_to_int = dict((name, number) for number, name in enumerate(i))
idx = sorted(name_to_int.values())

# vectorize token list
vectors = np.zeros((len(idx),max(idx) + 1))
vectors[np.arange(len(vectors)),idx] = 1

# convert from numpy array to tensor
vector_tensor =  th.from_numpy(vectors)

# initialize lists holding graphs
target_container = []
source_container = []

# make the target graph list
for num, item in enumerate(data["target_ast"], start = 0):
    edge_list = []
    tensor_list = []
    location = 1
    total = 0
    # start graph
    g = dgl.DGLGraph()
    # parse through current target ast
    input_1 = str(data["target_ast"][num]).replace("\\n","").replace("root","").replace("children","").replace("[","").replace("]","").replace("'","").replace(":","").replace(",","").replace("{","{ ").replace("}"," }").split()
    # build the graph
    fill()
    # make tuples for the edges and save in source and destination
    src, dst = tuple(zip(*edge_list))
    # make edges between nodes
    g.add_edges(src, dst)
    #g.add_edges(dst, src)
    # attach tensor values to nodes
    g.ndata['info'] = th.randn(total, len(name_to_int))

    # for every node in the tree
    for num, item in enumerate(tensor_list, start = 0):
        # attach appropriate tensor
        g.ndata['info'][num] = tensor_list[num]
    # add to the list
    target_container.append(g)
    
# make the source graph list
for num,item in enumerate(data["source_ast"], start = 0):
    edge_list = []
    tensor_list = []
    location = 1
    total = 0

    # start graph
    g = dgl.DGLGraph()
    # parse through current source ast
    input_1 = str(data["source_ast"][num]).replace("\\n","").replace("root","").replace("children","").replace("[","").replace("]","").replace("'","").replace(":","").replace(",","").replace("{","{ ").replace("}"," }").split()
    # build the graph
    fill()
    # make tuples for the edges and save in source and destination
    src, dst = tuple(zip(*edge_list))
    # make edges between nodes
    g.add_edges(src, dst)
    #g.add_edges(dst, src)
    # attach tensor values to nodes
    g.ndata['info'] = th.randn(total,len(name_to_int))

    # for every node in the tree
    for num, item in enumerate(tensor_list, start = 0):
        # attach appropriate tensor
        g.ndata['info'][num] = tensor_list[num]
    # add to the list
    source_container.append(g)
    
#print example graph
nx.draw_kamada_kawai(target_container[78].to_networkx(), with_labels=True)

