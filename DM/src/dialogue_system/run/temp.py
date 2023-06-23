import pickle
action = pickle.load(open('Data/action_set.p', 'rb'))
symptoms = pickle.load(open('label1/disease_symptom.p', 'rb'))
slot = pickle.load(open('Data/slot_set.p', 'rb'))
gl = pickle.load(open('Data/goal_set.p', 'rb'))
# P = []
# for j in gl['test']:
# 	P = P + [int(j['consult_id'])]
# P.sort()
# print(P)
# print(len(action))
# print('Actions : ',action)
# print(len(slot))
# print('\n slots: ',slot)
# print('Goal Sample len:',len(gl['test']))
#print('Goal:',gl)
#print(symptoms)
#key = list(symptoms.keys())
#print(key)
#print(list((symptoms[key[0]]['symptom']).keys())[0:5])
# l = [1,4,5,6,7,12,13,14,19]
# ALL = {}
# for i in range(1,10):
# 	f = 'label' + str(l[i-1]) + '/disease_symptom.p'
# 	disease_symptom = pickle.load(file=open(f, "rb"))
# 	diease = list(disease_symptom.keys())
# 	t = {}
# 	for j in diease:
# 		top5 = list((disease_symptom[j]['symptom']).keys())[0:5]
# 		t.update({j:top5})
#
# 	temp = {l[i-1]:t}
# 	ALL.update(temp)

# print(ALL)
# with open('GDTop5.p', 'wb') as handle:
#     pickle.dump(ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# print(len(ALL.keys()))


multimodal_dict ={'M23':'Edema','M61':'Cyanosis','M91':'Ulcer','M97':'Proptosis','M101':'Eye swelling'}

multimodal_symp = ['Edema','Cyanosis','Ulcer','Proptosis','Eye swelling']

P = {'Edema':['OTHER', 'M23', 'OTHER', 'M23', 'M23', 'M23', 'M23', 'M23', 'M23', 'M23'],'Cyanosis':['OTHER', 'OTHER', 'M61', 'M61', 'OTHER', 'OTHER', 'M61', 'OTHER', 'M61', 'OTHER'],'Ulcer':['M97','M97','M97','M97','M97','M97','M97','M97','M97','M97'],'Proptosis':['OTHER', 'M97', 'OTHER', 'OTHER', 'M97', 'M97', 'M97', 'M97', 'M97', 'OTHER'],'Eye swelling':['OTHER', 'M101', 'M101', 'M101', 'OTHER', 'M101', 'OTHER', 'OTHER', 'M101', 'M101']}

new_d = {}
for j in range(0,len(multimodal_dict)):
	temp = P[multimodal_symp[j]]
	temp2 = []
	for k in temp:
		if k!='OTHER':
			temp2=temp2+[multimodal_dict[k]]
		else:
			temp2 = temp2 + ['OTHER']
	new_d.update({multimodal_symp[j]:temp2})

print(new_d)

pickle.dump(new_d, open('MDD_DesneNetCNN.p', 'wb'))
