import json
import cPickle as pkl
import array

def getOneHotEncoding(data):
	words = data.split()
	oneHotEncoding = []
	for word in words:
		oneHotEncoding.append(dictionary[word])
	return oneHotEncoding
	

trainingSampleSize = 10000
testingSampleSize = trainingSampleSize + 2000
reviewFilePath = "../Electronics_5.json"
#reviewFilePath = "../20reviews.json"
#trainingSampleSize = 2
#testingSampleSize = trainingSampleSize + 1

print("Training sample size selected : "  + str(trainingSampleSize))
print("Testing sample size selected : "  + str(testingSampleSize - trainingSampleSize))
print("Reading file : " + reviewFilePath)
reviewsProcessed = 0

X_train=[]
y_train=[]
X_test=[]
y_test=[]
posCount = 1
negCount = 1

oneHotNumber = 1
dictionary = dict()
temp = 1

with open(reviewFilePath) as f:
	for line in f:
		jsonRead = json.loads(line)
		
		helpfulness = jsonRead['helpful']
		
		if(int(helpfulness[1]) == 0 or (float(helpfulness[0])/float(helpfulness[1]) >=0.31 and float(helpfulness[0])/float(helpfulness[1]) <= 0.9)):
			continue

		reviewsProcessed += 1
		review = jsonRead["reviewText"]		
		review = review.replace('\'', '').replace('.', ' ').replace('!', ' ').replace('?', ' ').replace('\"', ' ').replace(',', ' ').replace("&", ' and ').replace(';', ' ').lower()	
		words = review.split()
		for word in words:
			if(word not in dictionary):
				dictionary[word] = oneHotNumber
				oneHotNumber += 1

		if(float(helpfulness[0])/float(helpfulness[1]) > 0.5):
			temp += 1
			if(temp % 5 != 0):
				continue
			
			if(posCount <= trainingSampleSize):
				print("Positive Training Samples processed : " + str(posCount))
				posCount += 1				
				y_train.append(1)				
				X_train.append(getOneHotEncoding(review))
			elif(posCount <= testingSampleSize) :
				print("Positive Testing  Samples processed : " + str(posCount - trainingSampleSize))
				posCount += 1
				y_test.append(1)
				X_test.append(getOneHotEncoding(review))				
			
		else:
			
			if(negCount <= trainingSampleSize):		
				print("Negative Training Samples processed : " + str(negCount))
				negCount += 1		
				y_train.append(0)
				X_train.append(getOneHotEncoding(review))
			elif(negCount <= testingSampleSize) :
				print("Negative Testing  Samples processed : " + str(negCount - trainingSampleSize))
				negCount += 1
				y_test.append(0)
				X_test.append(getOneHotEncoding(review))
		if(posCount + negCount >= 2*(testingSampleSize+1)):
			break
	
print("**********************************")
print("Total Training samples : " +  str(len(X_train)))
print("Total Testing samples : " +  str(len(X_test)))
print("Vocabulary size : " +  str(len(dictionary)))
print("**********************************")	

fileName = "data/Electronics20000.pkl"
print("Saving data to pickle file : " + fileName)

f = open(fileName, 'wb')
pkl.dump((X_train, y_train, X_test, y_test), f, -1)
f.close()

		

#wf = open(test ,'w')
#wf.write(allData  )	
#wf.close()
