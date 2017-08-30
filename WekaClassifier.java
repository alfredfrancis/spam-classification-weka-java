import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.Evaluation;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.converters.ArffSaver;
import weka.classifiers.meta.FilteredClassifier;

import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;

// http://geekswithblogs.net/razan/archive/2011/11/08/creating-a-simple-sparse-arff-file.aspx
// http://weka.wikispaces.com/Programmatic+Use

public class WekaClassifier {

	private FilteredClassifier classifier;
	private Instances trainData;
	private Instances testData;
	private ArrayList<Attribute> fvWekaAttributes;

	WekaClassifier(){

		 //Initialize the FilteredClassifier
		 classifier = new FilteredClassifier();
		 
		 // Declare text attribute
		 Attribute attribute_text = new Attribute("text",(List<String>) null);

		 // Declare the label attribute along with its values
		 ArrayList<String> classAttributeValues = new ArrayList<String>();
		 classAttributeValues.add("spam");
		 classAttributeValues.add("ham");
		 Attribute classAttribute = new Attribute("label", classAttributeValues);

		 // Declare the feature vector
		 fvWekaAttributes = new ArrayList<Attribute>();
		 fvWekaAttributes.add(classAttribute);
		 fvWekaAttributes.add(attribute_text);

	}
	public Instances load (String filename)  throws IOException
	{
		 /* 
		    Create an empty training set
			name the relation “Rel”.
			set intial capacity of 10*
		*/	
		 Instances dataset = new Instances("Rel", fvWekaAttributes, 10);

		 // Set class index
		 dataset.setClassIndex(0);

		 // read text file, parse data and add to instance
		try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
		    for(String line; (line = br.readLine()) != null; ) {
		    	try{

		    		// split at first occurance of n no. of words
		        	String parts[] = line.split("\\s+",2);

		        	 // basic validation
		        	if (!parts[0].isEmpty() && !parts[1].isEmpty()){

		        	  DenseInstance row = new DenseInstance(2);
					  row.setValue(fvWekaAttributes.get(0), parts[0]);
					  row.setValue(fvWekaAttributes.get(1), parts[1]);	

					  // add row to instances
					  dataset.add(row);
		        	}
					// 
		    	}
		    	catch (ArrayIndexOutOfBoundsException e){
					System.out.println("invalid row");
				}

		    }

		}
		catch (IOException e){
			e.printStackTrace();
		}
		 return dataset;

	}

	public void prepare() throws Exception{
		trainData = load("dataset/train.txt");
		testData = load("dataset/test.txt");
	}

	public void transform(){

		// create the filter and set the attribute to be transformed from text into a feature vector (the last one)
		StringToWordVector filter = new StringToWordVector();
		filter.setAttributeIndices("last"); 

		classifier.setFilter(filter); 
		classifier.setClassifier(new NaiveBayesMultinomial());

	}
	public void fit() throws Exception{
		classifier.buildClassifier(trainData);
	}

	public String classify(String text) throws Exception  {

			Instances newDataset = new Instances("testdata", fvWekaAttributes, 1);
			newDataset.setClassIndex(0);
		
			DenseInstance newinstance = new DenseInstance(2);
			newinstance.setDataset(newDataset); 

			newinstance.setValue(fvWekaAttributes.get(1), text);

			double pred = classifier.classifyInstance(newinstance);

			System.out.println("===== Classified instance =====");
			System.out.println("Class predicted: " + trainData.classAttribute().value((int) pred));
			return trainData.classAttribute().value((int) pred);
	}

	public String evaluate() throws Exception{
		Evaluation eval = new Evaluation(testData);
		eval.evaluateModel(classifier, testData);
		System.out.println(eval.toSummaryString());
		return eval.toSummaryString();
	}


	public void saveArff(Instances dataset,String filename)   throws IOException{
		try
		{
			// initialize 
		   ArffSaver arffSaverInstance = new ArffSaver(); 
	       arffSaverInstance.setInstances(dataset); 
	       arffSaverInstance.setFile(new File(filename)); 
	       arffSaverInstance.writeBatch();	
		}
		catch (IOException e){
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws Exception{

		WekaClassifier wt = new WekaClassifier();
		wt.prepare();
		wt.transform();
		wt.fit();
		wt.evaluate();
		wt.classify("free foods");
	}
}