import os
import numpy as np
import shutil

def delete_result_files():
    path='/home/rahul/Desktop/mln_try2/train_original'
    for p in os.listdir(path):
        path1=path+'/'+p
        for video in os.listdir(path1):
            if video[-6:]=='RESULT':
                path2=path1+'/'+video
                for f in os.listdir(path2):
                    path3=path2+'/'+f
                    print(path3)
                    if os.path.isfile(path3):
                        os.remove(path3)
                    elif os.path.isdir(path3):
                        shutil.rmtree(path3)

def get_actual_affordance(train_str):
    temp1=train_str.split('\n')
    dictt={}
    objects=[]
    for a in temp1:
        a=a.strip()
        if a=='':
            continue
        #if a[:5]=='Affor' or a[:5]=='//Aff':
        if a[:5]=='Affor':
            aff_temp=a.split(',')[0].split('(')[1].strip()
            obj_temp=a.split(',')[1].split(')')[0].strip()
            dictt[obj_temp]=aff_temp
            objects.append(obj_temp)
    return dictt,objects

def get_result_affordance(result_str):
    temp1=result_str.split('\n')
    dictt={}
    prob_dictt={}
    for a in temp1:
        a=a.strip()
        if a=='':
            continue
        aff=a.split(',')[0].split('(')[1].strip()
        obj=a.split(',')[1].split(')')[0].strip()
        prob=a.split(')')[1].strip()
        prob=float(prob)
        if obj in dictt:
            if prob_dictt[obj]<prob:
                dictt[obj]=aff
                prob_dictt[obj]=prob
        else:
            dictt[obj]=aff
            prob_dictt[obj]=prob
    return dictt

def find_all_affordance(path):
    classes={}
    class_names=[]
    class_count=0
    for video in os.listdir(path):
        if video[-6:]=='RESULT':
            path1=path+'/'+video
            path2=path+'/'+video[:-6]+'Train'
            for f in os.listdir(path1):
                train_file_path=path2+'/'+f[:-7]+'-train.db'
                file1=open(train_file_path)
                train_str=file1.read()
                file1.close()
                temp=train_str.split('\n')
                train_str=[]
                for a in temp:
                    a=a.strip()
                    if a=='':
                        continue
                    #if a[:5]=='Affor' or a[:5]=='//Aff':
                    if a[:5]=='Affor':
                        aff_temp=a.split(',')[0].split('(')[1].strip()
                        if aff_temp not in classes:
                            class_names.append(aff_temp)
                            classes[aff_temp]=class_count
                            class_count+=1
    return classes,class_names,class_count

def evaluate(path,classes,class_names,class_count):
    result_track=np.zeros(shape=(class_count,class_count))
    for video in os.listdir(path):
        if video[-6:]=='RESULT':
            path1=path+'/'+video
            path2=path+'/'+video[:-6]+'Train'
            for f in os.listdir(path1):
                train_file_path=path2+'/'+f[:-7]+'-train.db'
                result_file_path=path1+'/'+f
                file1=open(train_file_path)
                train_str=file1.read()
                file1.close()
                file1=open(result_file_path)
                result_str=file1.read()
                file1.close()
                train_aff,objects=get_actual_affordance(train_str)
                result_aff=get_result_affordance(result_str)
                for o in objects:
                    result_track[classes[train_aff[o]]][classes[result_aff[o]]]+=1
    return result_track

def find_accuracy(class_names,result_track,classes,):
    total_correct=0
    tt=0
    for c in class_names:
        ind=classes[c]
        total=0
        correct=result_track[ind][ind]
        for i in range(result_track.shape[1]):
            total+=result_track[ind][i]
        tt+=total
        total_correct+=correct
        accuracy=correct*100/total
        print(str(ind)+' : '+c+' : '+str(accuracy)+'%')	
        #print('')

    total_accuracy=total_correct*100/tt
    print('Total : '+str(total_accuracy)+'%')

def train_queries():
    subactivities=['moving','placing','reaching','cleaning','opening','closing','pouring','drinking']
    affordances=['movable','placeable','containable','reachable','cleanable','cleaner','openable','closeable','pourto','pourable','drinkable']
    final_str=''
    flag2=1
    threshold=30
    for subactivity in subactivities:
        path='/home/rahul/Desktop/mln_try2/train_original/1'
        dest=''
        str1=''
        i=0
        path_str=''
        dict1={}
        flag=1
        count=0
        for video in os.listdir(path):
            path2=path+'/'+video
            if video[-5:]=='Train':
                for f in os.listdir(path2):
                    if f.split('-')[0]==subactivity:
                        path3=path2+'/'+f
                        file=open(path3)
                        data=file.read()
                        file.close()
                        afford_dict,objects=get_actual_affordance(data)
                        flag3=1
                        for obj in objects:
                            afford=afford_dict[obj]
                            if afford not in dict1:
                                dict1[afford]=0
                            dict1[afford]=dict1[afford]+1
                            if flag3 and dict1[afford]<=threshold:
                                print(subactivity,afford_dict)
                                if flag:
                                    path_str+=path2+'/'+f
                                    flag=0
                                else:
                                    path_str+=','+path2+'/'+f
                                count+=1
                                flag3=0
        print(subactivity,dict1)

        #f1=open(subactivity+'-train_filenames.db','w+')
        #f1.write("%s" % path_str)
        #f1.close()  
        path_str='../../alchemy/bin/learnwts -d -i affordancerules.mln -o '+subactivity+'-out.mln -t '+path_str+' -ne Affordance -multipleDatabases'
        #print(path_str)
        if flag2:
            final_str+=path_str
            flag2=0
        else:
            final_str+='\n'+path_str
    print(final_str)

def test_query():
    path='/home/rahul/Desktop/mln_try2/train_original/1'
    dest=''
    str1=''
    i=0
    path_str=''
    flag=1
    for video in os.listdir(path):
        path2=path+'/'+video
        path3=path+'/'+video[:-4]+'RESULT'
        if video[-4:]=='Test':
            for f in os.listdir(path2):
                subactivity=f.split('-')[0]
                if subactivity!='null':
                    if flag:
                        path_str+='../../alchemy/bin/infer -ms -i '+subactivity+'-out.mln -r '+path3+'/'+f[:-8]+'.result -e '+path2+'/'+f+' -q Affordance'
                        flag=0
                    else:
                        path_str+='\n'+'../../alchemy/bin/infer -ms -i '+subactivity+'-out.mln -r '+path3+'/'+f[:-8]+'.result -e '+path2+'/'+f+' -q Affordance'

    #f1=open('test_queries.db','w+')
    #f1.write("%s" % path_str)
    #f1.close()  
    print(path_str)
    #../../alchemy-2/bin/infer -ms -i moving-out.mln -r moving.result -e test/moving-test.db -q Affordance

path='/home/rahul/Desktop/mln_try2/train_original/1'

delete_result_files()

train_queries()

test_query()

classes,class_names,class_count=find_all_affordance(path)
print(classes)
result_track=evaluate(path,classes,class_names,class_count)

print()result_track)

find_accuracy(class_names,result_track,classes,)

