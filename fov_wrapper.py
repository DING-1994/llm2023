import numpy as np
import networkx as nx

from hrs_hot_file import hrs_hot_func

def neighbor_filter_obs(env, state_repre_flag):

    agent_num = env.agent_num
    n_act = env.n_actions
    all_onehot_obs = np.array(env.obs_onehot)
    onehot_obs = all_onehot_obs[:,:n_act]
    G = env.ee_env.G

    state = [0] * n_act
    pos_list = []

    # get all agent state and position
    for i, obs_i in enumerate(onehot_obs):
        edge_or_node = tuple([i for i, o in enumerate(obs_i) if o!=0])
        if len(edge_or_node)==1:
            node = edge_or_node[0]
            pos = {"type": "n", "pos": node}
            obs_i = np.array(obs_i)*agent_num
        else:
            edge = edge_or_node
            pos = {"type": "e", "pos": edge, "current_goal": env.current_goal[i], "obs": obs_i}
        state += obs_i
        pos_list.append(pos)
    # print("state", state)
    # print("pos_list", pos_list)

    neighbor_filter = calc_neighbor_filter(pos_list, G, state, n_act, agent_num)
    #print("neighbor_list", neighbor_filter)

    if state_repre_flag=="onehot_fov":
        obs = all_onehot_obs
    elif state_repre_flag=="heu_onehot_fov":
        obs = hrs_hot_func(env, env.obs)

    #print("ori_obs", obs)

    for i in range(agent_num):
        for j in range(n_act):
            #print(i,j,neighbor_filter[i][j])
            if neighbor_filter[i][j]==-1:
                obs[i][j] = neighbor_filter[i][j]

    return obs

def get_nodes_to_be_consideration(agent_pos, graph):
    if agent_pos["type"]=="n":
        start_node = agent_pos["pos"]
    elif agent_pos["type"]=="e":
        start_node = agent_pos["current_goal"]
    return start_node, list(nx.neighbors(graph, start_node))

# return [ (0 or -1) * n_nodes ] * agent_num
# if there is another agent -> -1
def calc_neighbor_filter(pos_list, graph, state, n_act, agent_num):
    neighbor_list = []

    for i in range(agent_num):
        pos_data = pos_list[i]
        start_node, target_nodes = get_nodes_to_be_consideration(pos_data, graph)
        c = [0]*n_act # (0 or -1) * n_nodes
        if pos_data["type"]=="e":
            if  state[start_node]>agent_num:
                c[start_node] = -1
        elif pos_data["type"]=="n":
            for node in target_nodes:
                if state[node]>=agent_num or (state[node]>0 and state[start_node]>agent_num):
                    c[node] = -1
        neighbor_list.append(c)

    return neighbor_list