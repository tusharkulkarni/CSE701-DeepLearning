import json
import cPickle as pkl
import array

def getOneHotEncoding(data):
	words = data.split()
	oneHotEncoding = []
	for word in words:
		oneHotEncoding.append(dictionary[word])
	return oneHotEncoding
	

trainingSampleSize = 20000
dataSize = 25000
reviewFilePath = "../Electronics_5.json"
positiveReviewSupressor = 0
#reviewFilePath = "../20reviews.json"
#trainingSampleSize = 2
#testingSampleSize = trainingSampleSize + 1

print("Training sample size selected : "  + str(trainingSampleSize))
print("Testing sample size selected : "  + str(dataSize - trainingSampleSize))
print("Reading file : " + reviewFilePath)


X_train=[]
y_train=[]
X_test=[]
y_test=[]
count = 0

oneHotNumber = 1
dictionary = dict()

with open(reviewFilePath) as f:
	for line in f:
		jsonRead = json.loads(line)

		helpfulness = jsonRead['helpful']		
		if(int(helpfulness[1]) == 0):
			continue

		helpful = float(helpfulness[0])/float(helpfulness[1]) 
		positiveReviewSupressor += 1

		if(helpful > 0.42 and positiveReviewSupressor % 5 != 0):
			continue

		review = jsonRead["reviewText"]		
		review = review.replace('\'', '').replace('.', ' ').replace('!', ' ').replace('?', ' ').replace('\"', ' ').replace(',', ' ').replace("&", ' and ').replace(';', ' ').lower()	
		words = review.split()
		count += 1	
		for word in words:
			if(word not in dictionary):
				dictionary[word] = oneHotNumber
				oneHotNumber += 1
		
		if(count <= trainingSampleSize):
			print("Training Samples processed : " + str(count))			
			y_train.append(round(helpful,1))				
			X_train.append(getOneHotEncoding(review))

		else:
			print("Testing  Samples processed : " + str(count - trainingSampleSize))
			y_test.append(round(helpful,1))
			X_test.append(getOneHotEncoding(review))				
		

		if(count >= dataSize):
			break
	
print("**********************************")
print("Total Training samples : " +  str(len(X_train)))
print("Total Testing samples : " +  str(len(X_test)))
print("Vocabulary size : " +  str(len(dictionary)))
print("**********************************")	

fileName = "data/Electronics25000.pkl"
print("Saving data to pickle file : " + fileName)

f = open(fileName, 'wb')
pkl.dump((X_train, y_train, X_test, y_test), f, -1)
f.close()

		

#wf = open(test ,'w')
#wf.write(allData  )	
#wf.close()
