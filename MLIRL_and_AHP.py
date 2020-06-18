import os
import numpy as np
import pandas as pd
import math
from itertools import product
import pickle
import matplotlib.pyplot as plt

def data_cleaning(data):
  temp=['reachable','movable','placeable','openable','cleaner','closeable','containable','drinkable','pourto']
  states={}
  for s in temp:
    states[s]=0
  new={}
  for a in data:
    for p in data[a]:
      temp2=[]
      for i in range(len(data[a][p])):
        temp3=[]
        for j in range(len(data[a][p][i])):
          pair=[]
          for k in range(len(data[a][p][i][j][0])):
            temp=data[a][p][i][j][0][k]
            if temp in states:
              s=temp
              aa=data[a][p][i][j][1]
              if aa=='null':
                print('null action present')
              pair=[s,aa]
              break
          if len(pair)!=0:
            temp3.append(pair)
        temp2.append(temp3)
      if a not in new:
        new[a]={}
      new[a][p]=temp2
  print('Data cleaning done.')
  return new

def enumerate_states_and_action(n_features,state_space,action_space):
 
  action_dict = {}
  reverse_action_dict = {}
  action_count = 0

  state_dict = {}
  reverse_state_dict = {}
  state_count = 0

  n_states=len(state_space)
  features = np.zeros((n_states, n_features))
  for i in range(n_states):
    state_dict[state_space[i]]=i
    reverse_state_dict[i]=state_space[i]
    features[i][i]=1
  for i in range(len(action_space)):
    action_dict[action_space[i]]=i
    reverse_action_dict[i]=action_space[i]
  print('Enumeration of states and action done.')
  return features,action_dict,reverse_action_dict,state_dict,reverse_state_dict

def split_train_test_data(data,train_percent):
  demos={}
  for act in data:
    temp=[]
    for p in data[act]:
      for i in range(len(data[act][p])):
        temp.append(data[act][p][i])
    demos[act]=temp

  n_demos=len(demos['arranging_objects'])
  n_test=int(n_demos*(1-train_percent))
  n_train=n_demos-n_test
  print('For each activity --> n_train='+str(n_train)+' , n_test='+str(n_test))
  train_demos={}
  test_demos={}
  for act in demos:
    test_demos[act]=demos[act][0:n_test]
    train_demos[act]=demos[act][n_test:]
    print(act, 'train:',len(train_demos[act]),'test:',len(test_demos[act]))
  return train_demos,test_demos

def print_demos(test_demos,train_demos):
  for act in test_demos:
    print(act)
    print('test')
    for demo in test_demos[act]:
      print(demo)
    print('train')
    for demo in train_demos[act]:
      print(demo)
    print()

def print_reward(r,state_space):
  for i in range(len(state_space)):
    print(state_space[i]+' : '+str(r[i]))
  print()

def preprocess_demonstration(demos,state_dict,action_dict):
  new_demos={}
  for act in demos:
    trajectories=demos[act]
    temp1=[]
    for trajectory in trajectories:
      temp2=[]
      for pair in trajectory:
        s=state_dict[pair[0]]
        a=action_dict[pair[1]]
        temp2.append([s,a])
      temp1.append(temp2)
    new_demos[act]=temp1
  return new_demos

def get_transition_matrix(demos, n_states, n_action):
  T = np.zeros((n_states,n_action,n_states))
  N = len(demos)
  for i in range(N):
    L = len(demos[i])
    if L<=1:
      continue
    for j in range(L-1):
      state = demos[i][j][0]
      action = demos[i][j][1]
      next_state = demos[i][j+1][0]
      T[state][action][next_state]+=1
  for i in range(n_states):
    for j in range(n_action):
      temp = 0
      for k in range(n_states):
        temp+=T[i][j][k]
      if not temp==0:
        T[i][j]/=temp
  return T


def value_iteration(r,V,delta_V, gamma, beta, K, threshold, T, demos, n_actions, features):
	
	n_states=features.shape[0]
	n_features=features.shape[1]
	Q=np.zeros((n_states,n_actions))
	delta_Q=np.zeros((n_states,n_actions,n_features))
	Z=np.zeros((n_states,1))
	delta_Z=np.zeros((n_states,n_features))
	pi=np.zeros((n_states,n_actions))
	delta_pi=np.zeros((n_states,n_actions,n_features))
	L=0
	delta_L=np.zeros((n_features,1))
	old_V=V
	old_delta_V=delta_V
	
	i=0
	change=float('inf')
	while i<K and change>threshold:
		i+=1
		for s in range(n_states):
			for a in range(n_actions):
				temp=0
				for sdash in range(n_states):
					temp+=gamma*T[s][a][sdash]*V[sdash]
				Q[s][a]=r[s]+temp
				for f in range(n_features):
					temp=0
					for sdash in range(n_states):
						temp+=gamma*T[s][a][sdash]*delta_V[sdash][f]
					delta_Q[s][a][f]=features[s][f]+temp

		for s in range(n_states):
			temp=0
			for a in range(n_actions):
				temp+=np.exp(beta*Q[s][a])
			Z[s]=temp
		
		for s in range(n_states):
			for f in range(n_features):
				temp=0
				for a in range(n_actions):
					temp+=beta*np.exp(beta*Q[s][a])*delta_Q[s][a][f]
				delta_Z[s][f]

		for s in range(n_states):
			for a in range(n_actions):
				pi[s][a]=np.exp(beta*Q[s][a])/Z[s]
				for f in range(n_features):
					delta_pi[s][a][f]=( ( beta*Z[s]*np.exp(beta*Q[s][a])*delta_Q[s][a][f] ) + ( np.exp(beta*Q[s][a])*delta_Z[s][f]  ) )/(Z[s]*Z[s])

		for s in range(n_states):
			temp=0
			for a in range(n_actions):
				temp+=pi[s][a]*Q[s][a]
			for f in range(n_features):
				temp=0
				for a in range(n_actions):
					temp+=(Q[s][a]*delta_pi[s][a][f])+(pi[s][a]*delta_Q[s][a][f])
		
		L=0
		for demo in demos:
			for step in demo:
				L+=np.log(pi[step[0]][step[1]])
		L/=len(demos)

		for f in range(n_features):
			temp=0
			for demo in demos:
				for step in demo:
					temp+=delta_pi[step[0]][step[1]][f]/pi[step[0]][step[1]]
			delta_L[f]=temp/len(demos)

		change=np.max(np.abs(old_V-V))		
		old_V=V
		old_delta_V=delta_V


	return L,delta_L

def mlirl(features, T, demos, gamma, alpha, beta, niterations, K, n_actions, threshold):
  n_states=features.shape[0]
  n_features=features.shape[1]
  w=np.random.uniform(size=(n_features,1))
  r=np.dot(features, w)
  for t in range(niterations):
    V=r
    delta_V=features
    #print(r,V,delta_V, gamma, beta, K, threshold, T, demos, n_actions, features)
    L, delta_L=value_iteration(r,V,delta_V, gamma, beta, K, threshold, T, demos, n_actions, features)
    w=w+(alpha*delta_L)
    r=np.dot(features, w)
  return r

def ahp(reward, n_features, n_states, state_space, test_demos, activities,optimal_r, optimal_correct, optimal_parameters, parameters):
	counter=0
	weights={}
	for act in reward:
		#print(act)
		r=reward[act]
		matrix=[[0]*n_states]*n_states
		matrix=np.zeros((n_states,n_states))
		dictt={}
		for i in range(n_states):
			for j in range(n_states):
				matrix[i][j]=r[j]/r[i]
			dictt[state_space[i]]=matrix[i]
    #print(matrix)
		criteria=pd.DataFrame({state_space[0]:matrix[0], state_space[1]:matrix[1], state_space[2]:matrix[2], state_space[3]:matrix[3], state_space[4]:matrix[4], state_space[5]:matrix[5], state_space[6]:matrix[6], state_space[7]:matrix[7], state_space[8]:matrix[8]}, index=state_space)
    #print(criteria)
		for s in state_space:
			criteria[s]/=sum(criteria[s])
    #print(criteria)
		criteria['r_sum']=criteria.movable + criteria.reachable + criteria.cleaner + criteria.openable + criteria.closeable + criteria.pourto + criteria.placeable + criteria.drinkable + criteria.containable
		criteria['r_avg/criteria_vector']=criteria.r_sum/len(state_space)
    #print(r)
    #print(criteria)
		weights[act]=criteria['r_avg/criteria_vector'].to_list()
		counter+=1
    #if counter>=1:
    #  break

	#test_demo=get_test_demo(d)
	
	for act in activities:
		correct=0
		for temp in test_demos[act]:
			#print('Actual activity : ',act)
			arr1=[]
			arr2=[]
			maxi_act=''
			maxi=-1000
			for a in activities:
				w=weights[a]
				temp2=0
				for i in range(len(temp)):
					temp2+=w[temp[i][0]]
				arr1.append(a)
				arr2.append(temp2)
				#print(maxi,temp2)
				if maxi<temp2:
					maxi=temp2
					maxi_act=a
			if maxi_act==act:
				correct+=1
				#print('Predicted activity : ',maxi_act, ' <--- Correct  ')
			#else:
				#print('Predicted activity : ',maxi_act)
			#fig = plt.figure()
			#ax = fig.add_axes([0,0,1,1])
			#ax.bar(arr1,arr2)
			#plt.show()
		if act not in optimal_correct:
			optimal_correct[act]=-1
		if optimal_correct[act]<correct:
			optimal_correct[act]=correct
			optimal_parameters[act]=parameters
			optimal_r[act]=reward[act]	

	return optimal_r, optimal_correct, optimal_parameters

def main(activities, n_states, n_action, state_space, features, train_demos, test_demos, alpha, beta, gamma, n_iterations, K, threshold, optimal_r, optimal_correct, optimal_parameters):

  reward={}
  np.random.seed(1)
  parameters=[alpha, beta, gamma, n_iterations, K, threshold]
  for act in train_demos:
    #np.random.seed(1)
    demos=train_demos[act]
    #T=get_transition_matrix(demos, n_states, n_action)
    demo1=demos
    demo2=test_demos[act]
    T=get_transition_matrix_2(demo1, demo2, n_states, n_action)
    r=mlirl(features, T, demos, gamma, alpha, beta, n_iterations, K, n_action, threshold)
    reward[act]=r
    print(act)
    print_reward(r,state_space)

  for act in reward:
    r=reward[act]
    reward[act]=np.asarray(r)

  optimal_r, optimal_correct, optimal_parameters=ahp(reward, n_features, n_states, state_space, test_demos, activities, optimal_r, optimal_correct, optimal_parameters, parameters)
  #print('Correct : '+str(n_correct))
  return optimal_r, optimal_correct, optimal_parameters


