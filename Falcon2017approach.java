/**
 * 
 */

//TODO : We can update all green and red inferred edges while asking the oracle itself..
//It will save us 2nd pass to find red and green inferred edges
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.RandomTree.Tree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


public class Falcon2017approach {
	String folder = "CarsCombined";//cora";
	PrintStream recallprint;
	HashMap<pair,Boolean> oracle = new HashMap<pair,Boolean>();
	static Random generator = new Random(1992);
	
	
	ArrayList<component> set_clusters = new ArrayList<component>();
	double theta = 0.30;
	int tau=0;
	int numIgnore=20;//Number of blocks to ignore
	int queries=0;
	double confidence = 0.95;
	double true_pos = 0,false_pos = 0;
	String pipelinetype = "iNoloop";
	double g_edges = 65000;//2588937;//3250373;//755388;//87512;//343524;//87512;//328411//65000;
	double g_to_r=0.0;
	double r_to_g=0.0;
	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	HashMap<Integer,String> recordList = new HashMap<Integer, String>();
	HashMap<Integer,Double> blockWeight = new HashMap<Integer, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();
	
	HashMap<String,Integer> numGreen = new HashMap<String, Integer>();
	HashMap<String,Integer> numRed = new HashMap<String, Integer>();
	
	HashMap<Integer,Integer> goldMap = new HashMap<Integer,Integer>();
	
	//ArrayList<HashMap<Integer,Double>> recordAdjacencyList = new ArrayList<HashMap<Integer,Double>>();
	
	HashMap<Integer, Double> expectedSize = new HashMap<Integer, Double>();
	
	
	
	
	
	HashMap<String,Integer> blockIndex = new HashMap<String, Integer>();
	HashMap<Integer,String> Index2Str = new HashMap<Integer, String>();
	ArrayList<ArrayList<Integer>> blockList = new ArrayList<ArrayList<Integer>>();
	ArrayList<double[]> blockSize = new ArrayList<double[]>();
	
	public ArrayList<double[]> samplePairs(int num){
		ArrayList<double[]> samplePairs = new ArrayList<double[]>();
		
		for(int a:recordList.keySet()) {
			for(int b:recordList.keySet()) {
				if(a<=b)
					continue;
				double[] tmp = {a,b};
				samplePairs.add(tmp);
			}
		}
		Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		Collections.shuffle(samplePairs,newgen);
		return new ArrayList<double[]>(samplePairs.subList(0, num));
	}
	
	public ArrayList<double[]> samplePairsFalcon(int num){
		
		int y=100;
		ArrayList<double[]> samplePairs = new ArrayList<double[]>();
		ArrayList<Integer> bcopy = new ArrayList<Integer>();
		ArrayList<Integer> acopy = new ArrayList<Integer>();

		for(int a:recordList.keySet()) {
			bcopy.add(a);
			acopy.add(a);
		}
		Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		
		Collections.shuffle(bcopy,newgen);
		for(int i=0;i<num*1.0/y;i++) {
			//System.out.println(i+"curr iteration");
			String curr = recordList.get(bcopy.get(i));
			String[] attributeList = curr.split(";",0);
			ArrayList<Integer> sharedRec=new ArrayList<Integer>();
			
		//	int recordId = Integer.parseInt(attributeList[0].trim());
			for (int k=1;k<attributeList.length;k++){
				String[] tokenList = attributeList[k].split(" ",0);
				for(int j=0;j<tokenList.length;j++) {
					if(tokenList[j].length()<=1)
						continue;
					String blockname= tokenList[j];
					ArrayList<Integer> blockContent= blockList.get(this.blockIndex.get(blockname));
					sharedRec.addAll(blockContent);
				}
			}
			Collections.sort(sharedRec);
			//System.out.println(i+"sharedrec"+sharedRec.size());
			ArrayList<double[]> blockorder= new ArrayList<double[]>();
			HashMap<Integer,Boolean> sharedBlock = new HashMap<Integer,Boolean>();
			int prev=-1;
			int currcount=0,currval=-1;
			for(int j=0;j<sharedRec.size();j++) {
				//System.out.println(sharedRec.get(j));
				if(currval != sharedRec.get(j)) {
					if(currval>-1) {
						double[] tmp = {currval, currcount};
						blockorder.add(tmp);
						sharedBlock.put(currval, true);
					}
					currval = sharedRec.get(j);
					currcount++;
				}else
					currcount++;
			}
			double[] tmp = {currval, currcount};
			blockorder.add(tmp);
			sharedBlock.put(currval, true);
			//System.out.println(i+"sharedrec"+blockorder.size()+" "+sharedBlock.keySet().size());

			 Collections.sort(blockorder, new Comparator<double[]>() {
					public int compare(double[] o1, double[] o2) {
						double s1 = o1[1]; double s2 = o2[1];
						if (s1 != s2)
							return (s1 > s2 ? -1 : 1);
						else
							return 0;
					}
				});
			 int iter=0,j=0;
			 
			for( iter=0;iter<Math.min(blockorder.size(),y*0.5 );j++) {
				if (bcopy.get(i)== blockorder.get(j)[0]) {
					continue;
				}
				double[] tmp1 = {bcopy.get(i), blockorder.get(j)[0]};
				samplePairs.add(tmp1);
				iter++;
			}
			//System.out.println(i+"sharedrec"+blockorder.size());

			Random newgen2 = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
			Collections.shuffle(acopy,newgen2);
			
			j=0;
			for(iter=0;iter<y-Math.min(blockorder.size(),y*0.5);j++){
				if(j>=acopy.size())
					break;
				if(sharedBlock.containsKey(acopy.get(j)))
					continue;
				else {
					double[] tmp1 = {bcopy.get(i), acopy.get(j)};
					samplePairs.add(tmp1);
				}	
				iter++;	
			}
			//System.out.println(i+"sharedrec"+blockorder.size());
		}
		return samplePairs;
	}
	
	
	public double jaccard (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		for(String a:s1.keySet()){
			if(s2.containsKey(a)){
				inter+=Math.min(s2.get(a), s1.get(a));
				union+=Math.max(s2.get(a), s1.get(a));
			}else
				union+=s1.get(a);
		}
		for(String a:s2.keySet()){
			if(s1.containsKey(a))
				continue;
			else
				union+=s2.get(a);		
		}
		return inter*1.0/union;
	}
	public HashMap<String,Integer> GetDict(String recordString) {
		recordString = recordString.toLowerCase( );
 		String[] tokenList = recordString.split(" ",0);
 		//System.out.println(recordString);
 		int val;
 		HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
 		
 		for (int i=0;i<tokenList.length;i++){
 			if(tokenList[i].length()<=1)
 				continue;
 			if(recordMap.containsKey(tokenList[i])){
 				val = recordMap.get(tokenList[i])+1;
 				recordMap.put(tokenList[i],val);
 				continue;//Avoiding double insertion of same element in the block
 			}
 			recordMap.put(tokenList[i],1);
 			ArrayList<Integer> blockContent;
		}
 		return recordMap;
	}
	public HashMap<String,Double> GetFeatures(int a, int b,int m){
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		String[] s1 = recordList.get(a).split(";",0);
		String[] s2 = recordList.get(b).split(";",0);
		for(int i=1;i<Math.min(m,s1.length);i++) {
			//TODO: change the key of the feature
			featMap.put(Integer.toString(i-1),jaccard(GetDict(s1[i]),GetDict(s2[i])));
		}
		featMap.put("id1",Double.valueOf(a));
		featMap.put("id2",Double.valueOf(b));
		if(goldMap.get(a).equals(goldMap.get(b)))
			featMap.put("class",1.0);
		else
			featMap.put("class",-1.0);
		return featMap;
		
	}
	public ArrayList<HashMap<String,Double>> genFunctions(ArrayList<double[]> samplePairs, int m) {
		//Generate m features
		ArrayList<HashMap<String,Double>> featList = new ArrayList<HashMap<String,Double>>();
		for(double[] sample :samplePairs) {
			HashMap<String,Double> featMap = GetFeatures((int)sample[0],(int)sample[1],m);
			featList.add(featMap);
		}
		return featList;
	}
	
	public boolean processConfidenceList(ArrayList<Double> confidenceLst, int window, int nconverged,int nhigh, int ndegrade, double epsilon) {
		ArrayList<Double> SmoothconfidenceLst = new ArrayList<Double>();
		if (confidenceLst.size() > window) {
			double sum=0;
			for (int i= 0;i<confidenceLst.size();i++) {
				sum +=confidenceLst.get(i);
				if (i>=window-1) {
					if (i>=window)
						sum -= confidenceLst.get(i-window);
					SmoothconfidenceLst.add(sum);
				}
			}
			
			if(SmoothconfidenceLst.size()>nconverged) {
				boolean violated=false;
				for(int j=SmoothconfidenceLst.size()-1; j>=SmoothconfidenceLst.size()-nconverged;j--) {
					for(int k = j-1;k>=SmoothconfidenceLst.size()-nconverged;k--) {
						if (Math.abs(SmoothconfidenceLst.get(j) - SmoothconfidenceLst.get(k)) > 2*epsilon) {
							violated=true;
							break;
						}
					}
					if (violated)
						break;
				}
				if (!violated)
					return true;
			}
			
			if(SmoothconfidenceLst.size()>nhigh) {
				boolean violated=false;
				for(int j=SmoothconfidenceLst.size()-1; j>=SmoothconfidenceLst.size()-nhigh;j--) {
					if (SmoothconfidenceLst.get(j)< 1-epsilon) {
						violated=true;
						break;
					}
				}
				if (!violated)
					return true;
			}
			double prevbest=-1,currbest=-1;
			boolean found=false;
			for (int i=0;i<SmoothconfidenceLst.size();i++) {
				if (i%ndegrade ==0) {
					if (prevbest>currbest)
						return true;
					prevbest=currbest;
					currbest=-1;
				}
				if (SmoothconfidenceLst.get(i)> currbest)
						currbest = SmoothconfidenceLst.get(i);
			}
			
				
		}
		return false;
	}
	public ArrayList<ArrayList<HashMap<String,Double[]>>> al_matcher(int m, ArrayList<HashMap<String,Double>> featureLst, ArrayList<Integer> AlreadyQueried, ArrayList<HashMap<String,Double>> V, boolean Blocking) throws Exception {
		//Already queried tells which answers from featurelst are already known to us and rest are the ones which we want to learn on
		ArrayList<Double> confidenceLst= new ArrayList<Double>();
		ArrayList<ArrayList<Instances>> ListofTrainingSetList = new ArrayList<ArrayList<Instances>>();
		ArrayList<ArrayList<Classifier>> ListofModelLst= new ArrayList<ArrayList<Classifier>>();
		ArrayList<ArrayList<ArrayList<HashMap<String,Double[]>>>> ListofruleLstlst= new  ArrayList<ArrayList<ArrayList<HashMap<String,Double[]>>>>();
		int window=5;
		int nconverged=20;
		int nhigh=3;
		int ndegrade=15;
		double epsilon=0.01;
		 
		while(true) {
			
			int numClassifiers=10;
			int featNumbers= featureLst.get(0).keySet().size()-2;
			int datasetSize=(int)(AlreadyQueried.size()*0.6);
			 ArrayList<Attribute> features = new ArrayList<Attribute>();
			 for(String s:featureLst.get(0).keySet()) {
				 if(s.equals("class"))
					 continue;
				 if(s.equals("id1") || s.equals("id2"))
					 continue;
				 Attribute tmp = new Attribute(s);
				 features.add(tmp);
			 }
			 // Declare the class attribute along with its values
			 FastVector classLabel = new FastVector(2);
			 classLabel.addElement("red");
			 classLabel.addElement("green");
			 Attribute ClassAttribute = new Attribute("class", classLabel);
			 
			 // Declare the feature vector
			 FastVector fvWekaAttributes = new FastVector(featNumbers);
			 for(Attribute s:features) {
				 fvWekaAttributes.addElement(s);
			 }
			 fvWekaAttributes.addElement(ClassAttribute);
			 
				
			ArrayList<Instances> TrainingSetList = new ArrayList<Instances>();
			ArrayList<Classifier> ModelLst= new ArrayList<Classifier>();
			 ArrayList<ArrayList<HashMap<String,Double[]>>> ruleLstlst= new  ArrayList<ArrayList<HashMap<String,Double[]>>>();
			for(int iter=0;iter<numClassifiers;iter++) {
				
			
			 Instances TrainingSet;
	
			 
			//////////////****************SIZEOF TRAINING SET
			 // Create an empty training set
			 TrainingSet = new Instances("Rel", fvWekaAttributes, datasetSize);
			 // Set class index
			 //System.out.println(fvWekaAttributes.size()+" "+featNumbers);
			 TrainingSet.setClassIndex(featNumbers-1);
			 
			 // Create the instance
			 
			 //////*****************Trainingset
			 
			 Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
			Collections.shuffle(AlreadyQueried,newgen);
				
			 for(int k=0;k<Math.min(AlreadyQueried.size(), datasetSize);k++) {
				 int recid = AlreadyQueried.get(k);
				 DenseInstance curr = new DenseInstance(featNumbers);
				 HashMap<String,Double>rowfeat= featureLst.get(recid);
				  for(int i=0;i<features.size();i++) {
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(i), rowfeat.get(Integer.toString(i)));
					
				 }
				 if(rowfeat.get("class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 TrainingSet.add(curr);
			 }
			 
			 
			  
			 Classifier cModel =new RandomTree();// new J48();//new Logistic();//new J48();
			 cModel.SetSeed((int)generator.nextDouble()*Integer.MAX_VALUE);
			 
			 cModel.buildClassifier(TrainingSet);
			 Tree t11 = cModel.GetRandomTree();
			 //System.out.println(t11.toString(0));
			 ArrayList<HashMap<String,Double[]>> rulelst = getRules(t11);
			/* System.out.println(rulelst.size()+"Number of rules ");
			 for(HashMap<String,Double[]> r : rulelst) {
				 System.out.println("New Rule ");
				 for(String s:r.keySet()) {
					 System.out.print(s+" "+r.get(s)[0]+" "+r.get(s)[1]+" ;;");
				 }
			 }*/
			 
			 ///////////***********Gotthe rules and store it with myself
			 ///////////*****Store the classifier also 
			 	ruleLstlst.add(rulelst);
			 	TrainingSetList.add(TrainingSet);
			 	ModelLst.add(cModel);
			}
			ListofTrainingSetList.add(TrainingSetList);
			ListofModelLst.add(ModelLst);
			ListofruleLstlst.add(ruleLstlst);
			 
			
			 
			 boolean classificationNeeded=true;
			 
			 ////////
			 double totalConfidence=0.0;
			 for(int i=0;i<V.size();i++) {
				 HashMap<String,Double>rowfeat= V.get(i);
				 DenseInstance curr = new DenseInstance(featNumbers);
				 
				  for(int j=0;j<features.size();j++) {
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(j), rowfeat.get(Integer.toString(j)));
				 }
				 if(rowfeat.get("class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 
				 ///Choose all classifiers
				 int g=0;
				 for(int j=0;j<numClassifiers;j++) {
					 curr.setDataset(TrainingSetList.get(j));
					 double[] fDistribution = ModelLst.get(j).distributionForInstance(curr);
					 //System.out.println(fDistribution[0]+" "+fDistribution[1]+" "+rowfeat.get("class"));
					 if(fDistribution[0]<0.5)
						 g++;
					 //double[]tmp = {i,Math.abs(fDistribution[0]-0.5)};
					 //confusingLst.add(tmp);
					 
				 }
				 double confidence;
				 if(g!=0 && g!=10)
					 confidence= 1 + 1*(g*0.1*Math.log(g*0.1) + (1-g*0.1)*Math.log(1-g*0.1));
				 else
					 confidence=1;
				 totalConfidence+=confidence;
			 }
			 totalConfidence = totalConfidence*1.0/V.size();
			 confidenceLst.add(totalConfidence);
			 
			 if (processConfidenceList(confidenceLst,window,nconverged,nhigh,ndegrade,epsilon))// int window, int nconverged,int nhigh, int ndegrade, double epsilon
				 break;
			 System.out.println(totalConfidence+" confidence"+V.size()+" "+confidenceLst.size()+" "+AlreadyQueried.size());
			 for(int i=0;i<confidenceLst.size();i++)
				 System.out.println("confidence" +i+" "+confidenceLst.get(i));

			 //Done by me to give it extra edge
			 if(!Blocking && totalConfidence>0.9)
				 break;
			 if (Blocking &&totalConfidence==1.0)
				 break;
			 
			 
			 
			 
			 if(classificationNeeded) {
				 
				 ArrayList<double[]> confusingLst = new ArrayList<double[]>();
				 for(int i=0;i<featureLst.size();i++) {
					 HashMap<String,Double>rowfeat= featureLst.get(i);
					 DenseInstance curr = new DenseInstance(featNumbers);
					 
					  for(int j=0;j<features.size();j++) {
							 curr.setValue((Attribute)fvWekaAttributes.elementAt(j), rowfeat.get(Integer.toString(j)));
						
					 }
					 if(rowfeat.get("class")>0)
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
					 else
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
					 
					 ///Choose all classifiers
					 int g=0;
					 for(int j=0;j<numClassifiers;j++) {
						 curr.setDataset(TrainingSetList.get(j));
						 double[] fDistribution = ModelLst.get(j).distributionForInstance(curr);
						 //System.out.println(fDistribution[0]+" "+fDistribution[1]+" "+rowfeat.get("class"));
						 if(fDistribution[0]<0.5)
							 g++;
						 
						 //double[]tmp = {i,Math.abs(fDistribution[0]-0.5)};
						 //confusingLst.add(tmp);
						 
					 }
					 double entropy;
					 if(g!=0 && g!=10)
						 entropy= -1*(g*0.1*Math.log(g*0.1) + (1-g*0.1)*Math.log(1-g*0.1));
					 else
						 entropy=0;
					 double[]tmp = {i,Math.abs(g*0.1-0.5),entropy};
					 confusingLst.add(tmp);
					 
				 }
				 Collections.sort(confusingLst, new Comparator<double[]>() {
						public int compare(double[] o1, double[] o2) {
							double s1 = o1[2]; double s2 = o2[2];
							if (s1 != s2)
								return (s1 > s2 ? -1 : 1);
							else
								return 0;
						}
					});
				 
				 
				 
				 
				 int newexamples = 20;
				 int p=100;
				 double total=0.0;
				 int i=0;
				 for(int iter=0;i<p;iter++) {
					 if(AlreadyQueried.contains(((int)confusingLst.get(iter)[0])))
							 continue;
					 total+=confusingLst.get(i)[2];
					 i++;
				 }
				 ArrayList<Integer> newExamples = new ArrayList<Integer>();
				 for( i=0;i<newexamples;i++) {
					 double val = generator.nextDouble();
					 double curr=0.0;
					 int j=0;
					 for(int iter=0;j<p;iter++) {
						 if(newExamples.contains((int)confusingLst.get(iter)[0]) || AlreadyQueried.contains((int)confusingLst.get(iter)[0]))
							 continue;
						 else {
							 if(curr<val && curr + confusingLst.get(j)[2]>=val ) {
								 
								 pair tmp = new pair(featureLst.get((int)confusingLst.get(iter)[0]).get("id1").intValue(),featureLst.get((int)confusingLst.get(iter)[0]).get("id2").intValue());
								 pair tmp1 = new pair(featureLst.get((int)confusingLst.get(iter)[0]).get("id2").intValue(),featureLst.get((int)confusingLst.get(iter)[0]).get("id1").intValue());
								 if(!queried_edge_map.containsKey(tmp)) {
									 queries++;
									 queried_edge_map.put(tmp, goldMap.get(tmp.x).equals(goldMap.get(tmp.y)));
									 queried_edge_map.put(tmp1, goldMap.get(tmp1.x).equals(goldMap.get(tmp1.y)));
									 System.out.println("queries is "+queries);
								 }
								 //System.out.println(newExamples.size()+"******addng here"+" "+queried_edge_map.keySet().size());
								 newExamples.add((int)confusingLst.get(iter)[0]);
								 total-=confusingLst.get(iter)[2];
								 break;
							 }
							 else {
								 curr +=confusingLst.get(iter)[2];
							 }
						 }
							j++; 
					 }
				 }
				 AlreadyQueried.addAll(newExamples);
				 System.out.println(newExamples.size()+" ***************new ecample size"+AlreadyQueried.size()+" "+queried_edge_map.keySet().size());
			 }else {
				 break;
			 }
			 
			 //for(int i=0;i<numreTrain;i++) {
				 //Ask these from crowd and add it into already queried and loop back
				 
			// }
			
			 

		}
		
		return ListofruleLstlst.get(ListofruleLstlst.size()-1);
		//TODO:ALMATCHER  CONVERGENCE CHECK
		
		 
	}
	
	public ArrayList<HashMap<String,Double[]>> getRules (Tree t) throws Exception {
		ArrayList<HashMap<String,Double[]>> lst = new ArrayList<HashMap<String,Double[]>>();
		if(t.m_Attribute==-1) {
			HashMap<String,Double[]> ruleMap = new HashMap<String,Double[]>();
			String color = t.leafString().trim().split(" ",0)[1];
			Double[] val; 
			if(color.equals("red")) {
				val= new Double[] {0.0,-1.0};
			}else {
				val= new Double[] {0.0,1.0};
			}
			//System.out.println( t.getLocalModel().dumpLabel(0, t.getTrainingData())+" "+color+"color"+t.getLocalModel().leftSide(t.getTrainingData()));//+" "+t.getLocalModel().leftSide(t.getTrainingData()).split(" ",0)[1]);
			ruleMap.put("class", val);
			
			lst.add(ruleMap);
			return lst;
			
		}else {
			System.out.println(t.m_SplitPoint+" "+t.m_Attribute);
			//text.append(m_Info.attribute(m_Attribute).name() + " < "
		      //      + Utils.doubleToString(m_SplitPoint, getNumDecimalPlaces()));
		     
			System.out.println(t.m_Successors.length);
			ArrayList<HashMap<String,Double[]>>  o1 =  getRules(t.m_Successors[0]);
			
			ArrayList<HashMap<String,Double[]>>  o2 =  getRules(t.m_Successors[1]);
			
			Double[] l = {-1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o1) {
				map.put(t.m_Attribute+";"+Math.random(),l);
				lst.add(map);
			}
			
			//System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2]);
			Double[] r = {1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o2) {
				map.put(t.m_Attribute+";"+Math.random(),r);
				lst.add(map);
			}
			
			return lst;
		}
	
		//return ruleMap;
	} 
	public boolean CheckNo(HashMap<String,Double[]> rule) {
		if(rule.get("class")[1]>0)
			return false;
		else
			return true;
	}
	public boolean satisfy(HashMap<String,Double> feat, HashMap<String,Double[]> rule) {
		
		for(String s:rule.keySet()) {
			 if(s.trim().equals("class"))
				 continue;
			 String s1 = s.split(";",0)[0];
			 //System.out.println(s+" "+s1+" "+rule.get(s)[0]+" "+rule.get(s)[1]);
			 if(feat.get(s1.trim())  <= rule.get(s)[1] && rule.get(s)[0]<0 )
				 continue;
			 else if (feat.get(s1.trim())  > rule.get(s)[1] && rule.get(s)[0]>0)
				 continue;
			 else {
				 //System.out.println(feat+" "+s+" "+s1);
				 return false;
			 }
			 //System.out.print(s+" "+r.get(s)[0]+" "+r.get(s)[1]+" ;;");
		 }
		
		return true;
	}
	public  ArrayList<HashMap<String,Double[]>> rulesEval(ArrayList<HashMap<String,Double[]>> rulelst, ArrayList<HashMap<String,Double>> featureLst, ArrayList<Integer> AlreadyQueried) {
		//Check if the rule is a no based rule
		//Calculate coverage of the rule over S
		//Calculate T
		//
		ArrayList<double[]> ruleorder=new ArrayList<double[]>();
		for(int j=0;j<rulelst.size();j++) {
			HashMap<String,Double[]> rule = rulelst.get(j);
			if(CheckNo(rule)) {
				int coverage=0;
				int pos=0;
				for(int i=0;i<featureLst.size();i++) {
					HashMap<String,Double> row = featureLst.get(i);
					if(satisfy(row,rule)) {
						coverage++;
						if(AlreadyQueried.contains(i)) {
							if(row.get("class")>0)
								pos++;
						}
					}	
				}
				//Add to a list
				double[]tmp = {j,1-pos*1.0/coverage};
				ruleorder.add(tmp);
			}
		}
		 Collections.sort(ruleorder, new Comparator<double[]>() {
				public int compare(double[] o1, double[] o2) {
					double s1 = o1[1]; double s2 = o2[1];
					if (s1 != s2)
						return (s1 > s2 ? -1 : 1);
					else
						return 0;
				}
			});
		
		int k=20;
		 int j=0;
		 ArrayList<HashMap<String,Double[]>> finalRules= new ArrayList<HashMap<String,Double[]>>();
		 System.out.println(ruleorder.size());
		 for(int jiter=0;j<k;jiter++) {
			 System.out.println(jiter+" "+ruleorder.size());
			 if(jiter>=ruleorder.size())
				 break;
			 HashMap<String,Double[]> rule  = rulelst.get((int)ruleorder.get(jiter)[0]);
			 ArrayList<Integer> ruleSatisfier= new ArrayList<Integer>();
			 for(int i=0;i<featureLst.size();i++) {
					HashMap<String,Double> row = featureLst.get(i);
					if(satisfy(row,rule)) {
						ruleSatisfier.add(i);
					}	
				}
			System.out.println("satisfies rules"+ruleSatisfier.size()+" "+featureLst.size());
			for(String s:rule.keySet())
				System.out.print(s+" "+rule.get(s)[1]+" ;;");
			
			ArrayList<Integer> X = new ArrayList<Integer>();
			int g=0;
			boolean added=false;
			while(true) {
				Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
				Collections.shuffle(ruleSatisfier,newgen);
				int b=20;
				int i=0;
				boolean exit=false;
				for(int iter=0;i<b;iter++) {
					if(iter>=ruleSatisfier.size())
						break;
					if(X.size()==ruleSatisfier.size()) {
						exit=true;
						break;
					}
					if(X.contains(ruleSatisfier.get(iter)))
						continue;
					X.add(ruleSatisfier.get(iter));
					pair p = new pair((featureLst.get(ruleSatisfier.get(iter)).get("id1").intValue()),(int)(featureLst.get(ruleSatisfier.get(iter)).get("id2").intValue()));
					 
					 pair tmp1 = new pair((featureLst.get(ruleSatisfier.get(iter)).get("id2").intValue()),(int)(featureLst.get(ruleSatisfier.get(iter)).get("id2").intValue()));
					 if(!queried_edge_map.containsKey(p)) {
						 queries++;
						 queried_edge_map.put(p, goldMap.get(p.x).equals(goldMap.get(p.y)));
						 queried_edge_map.put(tmp1, goldMap.get(tmp1.x).equals(goldMap.get(tmp1.y)));
						 System.out.println("queries is "+queries);
					 }
					if(goldMap.get(p.x).equals(goldMap.get(p.y)))
						g++;
					i++;
				}
				double p = (X.size()-g)*1.0/X.size();
				int m=ruleSatisfier.size();
				int n=X.size();
				double eps= 1.96*Math.sqrt(p*(1-p)*(m-n)*1.0/(n*(m-1)));

				if(exit)
					break;
				if(p>0.95 && eps <= 0.05) {
					added=true;
					break;
				}else if (p+eps < 0.95 || (eps<0.05 && p<0.95)) {
					//System.out.println("not enough confidence"+ruleSatisfier.size()+" "+X.size());
					break;
				}
				else
					continue;
				
			}
			if(added) {
				finalRules.add(rule);
				j++;
			}
			 //Evaluate the rule
			//Return the rule if good
		 }
		 return finalRules;
	}
	public boolean checkSame(HashMap<String,Double[]> r1, HashMap<String,Double[]> r2) {
		for(String s:r1.keySet()) {
			String key = s.split(";",0)[0];
			double val = r1.get(s)[1];
			double dir = r1.get(s)[0];
			boolean found=false;
			for(String s1:r2.keySet()) {
				if(s1.split(";",0)[0].equals(key) && val ==r2.get(s1)[1] && dir == r2.get(s1)[0]) {
					found=true;
					break;
				}
			}
			if(!found)
				return false;
		}
		
		for(String s:r2.keySet()) {
			String key = s.split(";",0)[0];
			double val = r2.get(s)[1];
			double dir = r2.get(s)[0];
			boolean found=false;
			for(String s1:r1.keySet()) {
				if(s1.split(";",0)[0].equals(key) && val ==r1.get(s1)[1] && dir == r1.get(s1)[0]) {
					found=true;
					break;
				}
			}
			if(!found)
				return false;
		}
		
		
		return true;
	}
	public ArrayList<HashMap<String,Double[]>> UniqRules (ArrayList<HashMap<String,Double[]>> rulelst){
		ArrayList<HashMap<String,Double[]>> uniqrulelst = new ArrayList<HashMap<String,Double[]>>();
		for (int i=0;i<rulelst.size();i++) {
			boolean same=false;
			HashMap<String,Double[]> rule = rulelst.get(i);
			for(int j=0;j<uniqrulelst.size();j++) {
				if(checkSame(rule,uniqrulelst.get(j))) {
					same=true;
					break;
				}
			}
			if(!same) {
				uniqrulelst.add(rule);
			}
		}
		return uniqrulelst;
	}
	
	public boolean satisfyRule(HashMap<String,Double[]> rule, HashMap<String,Double> feat) {
		// boolean holds=true;
		//System.out.println(feat);
		for(String s:rule.keySet()) {
			 if(s.trim().equals("class"))
				 continue;
			 String s1 = s.split(";",0)[0];
			 //System.out.println(s);
			 if(feat.get(s1.trim())  <= rule.get(s)[1] && rule.get(s)[0]<0 )
				 continue;
			 else if (feat.get(s1.trim())  > rule.get(s)[1] && rule.get(s)[0]>0)
				 continue;
			 else {
				 //System.out.println(feat+" "+s+" "+s1);
				 return false;
			 }
			 //System.out.print(s+" "+r.get(s)[0]+" "+r.get(s)[1]+" ;;");
		 }
		
		return true;
	}
	public double queryPair( ArrayList<ArrayList<HashMap<String,Double[]>>> randomForest,HashMap<String,Double> feat) {
		double confidence=0.0;
		int out=0;
		for(ArrayList<HashMap<String,Double[]>> tree: randomForest) {
			for(HashMap<String,Double[]> rule:tree) {
				if (satisfyRule(rule,feat)) {
					//System.out.println("satisfied"+" "+rule.get("class")[0]+" "+rule.get("class")[1]);
					out+=rule.get("class")[1];
				}
			}
		}
		return out*1.0/randomForest.size();
	}
	public void processBlocks() throws Exception{
		
		//Write the different operators and subroutines here along with features
		//Maintain a global list of queried edges so that I can use them to verify that these are all I know
		int numSample=8000;
		ArrayList<double[]> pairs = samplePairsFalcon(numSample);
		
		ArrayList<double[]> Validation = samplePairsFalcon((int)(0.03*numSample));
		
		System.out.println("pairs "+pairs.size());
		
		ArrayList<pair> green =new ArrayList<pair>(),red=new ArrayList<pair>();
		ArrayList<Integer> AlreadyQueried =  new ArrayList<Integer>();
		
		
		for(int i:recordList.keySet()) {
			for(int j:recordList.keySet()) {
				if(i>j) {
					pair p=new pair(i,j);
					if(goldMap.get(i).equals(goldMap.get(j)))
						green.add(p);
					else
						red.add(p);
				}
			}
		}
		 Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		Collections.shuffle(green,newgen);
			
		newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		Collections.shuffle(red,newgen);
		for(int i=0;i<2;i++) {
			double[]tmp = {green.get(i).x,green.get(i).y};
			pairs.add(tmp);
			AlreadyQueried.add(pairs.size()-1);
			
			double[]tmp1 = {red.get(i).x,red.get(i).y};
			pairs.add(tmp1);
			AlreadyQueried.add(pairs.size()-1);
		}
		
		ArrayList<HashMap<String,Double>> featvec = genFunctions(pairs,10);
		
		ArrayList<HashMap<String,Double>> V = genFunctions(Validation,10);

		
		System.out.println(featvec.size()+" "+AlreadyQueried.size()+" "+V.size()+" "+Validation.size());
		
		
		ArrayList<ArrayList<HashMap<String,Double[]>>> randomForest = al_matcher(10,  featvec,  AlreadyQueried, V,false); 
		
		ArrayList<HashMap<String,Double[]>> rulelst = new ArrayList<HashMap<String,Double[]>>();
		for(int i=0;i<randomForest.size();i++) {
			ArrayList<HashMap<String,Double[]>>tree = randomForest.get(i);
			rulelst.addAll(tree);
		}
		System.out.println("Got a matcher" +rulelst.size()+" "+queries+" "+featvec.size());
		//Get unique rules
		ArrayList<HashMap<String,Double[]>> Uniqrulelst = UniqRules( rulelst);
		
		 ArrayList<HashMap<String,Double[]>> finalRules = rulesEval( Uniqrulelst,  featvec,  AlreadyQueried);
		System.out.println(finalRules.size()+"final size"+" "+Uniqrulelst.size()+" "+rulelst.size());
		for(HashMap<String,Double[]> r : rulelst) {
			for(String s:r.keySet())
				System.out.print(s+" "+r.get(s)[1]+" ;;");
			System.out.println("dsjkl");
		}
		
		
		/////Get unique rules
		ArrayList<HashMap<String,Double>> candidatevec = new ArrayList<HashMap<String,Double>>();
		int g=0;
		for(int a:recordList.keySet()) {
			for(int b:recordList.keySet()) {
				if(a>b) {
					HashMap<String,Double> feat = GetFeatures( a,  b,10);
					boolean removed=false;
					for(HashMap<String,Double[]> r:finalRules) {
						if(satisfy(feat,r)) {
							removed=true;
							break;
						}
						break;
					}
					if(!removed) {
						candidatevec.add(feat);
						if(goldMap.get(a).equals(goldMap.get(b)))
							g++;
					}else {
						;//if(goldMap.get(a).equals(goldMap.get(b)))
							//g++;
					}
				}
				
			}
		}
		System.out.println(queries+" going to train for "+candidatevec.size());
		
		Validation = samplePairsFalcon((int)Math.min(recordList.keySet().size(),(0.03*candidatevec.size())));
		 randomForest = al_matcher(10,  candidatevec,  AlreadyQueried, V,false); 
		 System.out.println(queries+" "+candidatevec.size()+ " "+V.size()+" "+g);
		
		 g=0;
		 int fg=0;
		for(HashMap<String,Double> feat : candidatevec) {
			int a=feat.get("id1").intValue();
			int b=feat.get("id2").intValue();
			if(goldMap.get(a).equals(goldMap.get(b)) && queryPair(randomForest,feat)>=0.5 )
				g++;
			else if(!goldMap.get(a).equals(goldMap.get(b)) && queryPair(randomForest,feat)>=0.5 )
				fg++;
			
			
			//System.out.println(queryPair(randomForest,feat ));
		}
		System.out.println("greesn is "+g+" "+fg);
		return;
	
	}
	
	
	
	public static void main(String[] args) throws Exception{
    		
    		Falcon2017approach FF = new Falcon2017approach();
  
    		String recordFile     = FF.folder+"/recordsStructured.txt";
    		System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);

    		Scanner scanner = new Scanner(new File(recordFile));
    		HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
    		String[] attributeList ;
    		String line;
    		int val;
    		
    		while(scanner.hasNextLine()){
    			line= scanner.nextLine();
    			line = line.toLowerCase( );
    			//System.out.println(line);
    			attributeList = line.split(";",0);
    			//ArrayList<String> record = new ArrayList<String>();

    			HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
    			int recordId = Integer.parseInt(attributeList[0].trim());
    			//if(!largest.contains(recordId))

    			for (int i=1;i<attributeList.length;i++){
    				String[] tokenList = attributeList[i].split(" ",0);
    				for(int j=0;j<tokenList.length;j++) {
    					if(tokenList[j].length()<=1)
    						continue;
    					if(recordMap.containsKey(tokenList[j])){
    						val = recordMap.get(tokenList[j])+1;
    						recordMap.put(tokenList[j],val);
    						continue;//Avoiding double insertion of same element in the block
    					}
    					recordMap.put(tokenList[j],1);
    					
    					ArrayList<Integer> blockContent;
    					if(tmpblockList.containsKey(tokenList[j])){
    						blockContent = tmpblockList.get(tokenList[j]);
    					}
    					else
    						blockContent = new ArrayList<Integer>();

    					blockContent.add(recordId);
    					tmpblockList.put(tokenList[j], blockContent);
    				}
    			}
    			FF.recordList.put(recordId, line);
    		}
    		int loc=0;
    		for (String s:tmpblockList.keySet()){
    			FF.blockList.add(tmpblockList.get(s));
    			FF.blockIndex.put(s,loc);
    			FF.Index2Str.put(loc, s);
    			double[] tmp = {loc,tmpblockList.get(s).size()};
    			FF.blockSize.add(tmp);
    			loc+=1;
    		}
   	 System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
   	 System.gc();
   	 System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
   	 	
   	 
   	 	String goldFile     = FF.folder+"/gold.txt";
	
	 	Scanner goldScanner = new Scanner(new File(goldFile));
	 	while(goldScanner.hasNextLine()){
   	 		 line= goldScanner.nextLine();
   	 		//System.out.println(line);
   	 		String[] goldtokens = line.split(" ",0);
   	 		int u = Integer.parseInt(goldtokens[0]);
   	 		int id = Integer.parseInt(goldtokens[1]);
   	 		FF.goldMap.put(u, id);
   	 		//System.out.println(u+" "+id+" "+FF.goldMap.get(5));
	 	}
	 	int id = FF.recordList.keySet().size();
	 	for(int i:FF.recordList.keySet()){
	 		if(FF.goldMap.containsKey(i))
	 			continue;
	 		else{
	 			FF.goldMap.put(i, id);
	 			id++;
	 		}
	 	}
	 	
	 	System.out.println(" "+FF.goldMap.get(6));
   	 	//FF.printBlockStats();
   	 
   	 	FF.processBlocks();
   	 	System.out.println("here"+" "+FF.recordList.size());
    	
    }
    	
}

