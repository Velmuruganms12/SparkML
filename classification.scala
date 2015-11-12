########Manipulation les données##########
// On veut travailler avec un fichier qui contient pas une entête
//On va donc se déplacer dans le dossier qui contient les données et exécuter la commande suivante 
// La sortie est un nouveau fichier train_noheader.tsv
sed 1d train.tsv > train_noheader.tsv
//On doit lancer spark maintenant pour pourvoir tester la librairie MLLib 
//Il faut se déplacer dans le dossier d'installation de Spark pour lancer la commande suivante
./bin/spark-shell

//On va créer maintenant des RDD et appliquer une transformation (faire un split) 
val rawData = sc.textFile("train_noheader.tsv")
val records = rawData.map(line => line.split("\t"))
records.first


########Transformation des données##########
//On doit importer les librairies nécessaires pour pouvoir utiliser les algorithmes de MLLib 
//Les modèles de classification dans spark doivent implémenter LabeledPoint 
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

//On va maintenant remplace le caractère (") par null et remplacer aussi les valeurs représentées par (?) par (0)
val data = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
	LabeledPoint(label, Vectors.dense(features))
}

data.cache
val nbData= data.count
//On doit avoir comme sortie nbData:long=7395

//On a aussi des données négatives qu'on va remplacer par (0) pour pouvoir utiliser le modèle Naive Bayes  

val nbData = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
	LabeledPoint(label, Vectors.dense(features))
}


############Créer le modèle de Regression logistique##############
//// Modèle Regression logistique
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
val numIterations = 10
val maxTreeDepth = 5
val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

############Créer le modèle SVM##############
val svmModel = SVMWithSGD.train(data, numIterations)

############Créer le modèle Naive Bayes##############
val nbModel = NaiveBayes.train(nbData)

############Créer le modèle Arbre de decision##############
val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

############Générer les prédictions##############
//Générer des prédiction pour un seul individus des données 
val dataPoint = data.first
val prediction = lrModel.predict(dataPoint.features)
val trueLabel = dataPoint.label


//Générer des prédictions pour tout le RDD Data qu'on a crée
val predictions = lrModel.predict(data.map(lp => lp.features))
predictions.take(5)


###########Evaluer la performance du modèle crée##############
//Dans la formule suivante, on va tester l'indicateur "Prediction accuracy and error" pour le modèle Regression logistique
val lrTotalCorrect = data.map { point =>
  if (lrModel.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracy = lrTotalCorrect / numData


//Taux d'exactitude pour les autres modèles
val svmTotalCorrect = data.map { point =>
  if (svmModel.predict(point.features) == point.label) 1 else 0
}.sum
val nbTotalCorrect = nbData.map { point =>
  if (nbModel.predict(point.features) == point.label) 1 else 0
}.sum
val dtTotalCorrect = data.map { point =>
  val score = dtModel.predict(point.features)
  val predicted = if (score > 0.5) 1 else 0 
  if (predicted == point.label) 1 else 0
}.sum
val svmAccuracy = svmTotalCorrect / numData
val nbAccuracy = nbTotalCorrect / numData
val dtAccuracy = dtTotalCorrect / numData


############Améliorer la performance des modèles##############
//Standardization des variables
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val vectors = data.map(lp => lp.features)
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()
println(matrixSummary.mean)
println(matrixSummary.min)
println(matrixSummary.variance)

//Use the StandardScaler for the standardization
import org.apache.spark.mllib.feature.StandardScaler
val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

// Construire le modèle de regression logistique sur les données normalisées
val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
val lrTotalCorrectScaled = scaledData.map { point =>
  if (lrModelScaled.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaled = lrTotalCorrectScaled / numData



