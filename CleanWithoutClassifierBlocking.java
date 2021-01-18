/**
 * 
 */

//TODO : We can update all green and red inferred edges while asking the oracle itself..
//It will save us 2nd pass to find red and green inferred edges
import java.io.BufferedReader;

import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.stream.Collectors;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.RandomTree.Tree;
import weka.classifiers.trees.j48.ClassifierTree;


public class CleanWithoutClassifierBlocking {
	Random rand = new Random(1991);
	String folder = "citations";//songs";//"citations";//"songs";
	PrintStream recallprint;
	Classifier cModel;
	ArrayList<HashMap<String,Double[]>> ModelRules;
	ArrayList<ArrayList<HashMap<String,Double[]>>> ModelRuleslst;
	HashMap<pair,Boolean> oracle = new HashMap<pair,Boolean>();
	HashMap<pair,Boolean> training;
	ArrayList<component> set_clusters = new ArrayList<component>();
	double theta = 0.3;
	int tau=0;
	int N=0;
	int numIgnore=10;//Number of blocks to ignore
	int queries=0;
	double confidence = 0.95;
	double true_pos = 0,false_pos = 0;
	String pipelinetype = "pM3loop";
	double g_edges = 65000;//343524;//87512;//328411//65000;
	double g_to_r=0.0;
	double r_to_g=0.0;
	boolean useClassifier=false;
	int mergedBlocks=-1;
	boolean classifierExists=false;
	FastVector fvWekaAttributes;
	int featNumbers= 9;
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap1 = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap2 = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();

	
	HashMap<String, ArrayList<Integer>> blockQueries = new HashMap<String,ArrayList<Integer>>();
	HashMap<String, Integer> PrevQueries = new HashMap<String,Integer>();

	Set<Integer> processed_nodes = new HashSet<Integer>();
	HashMap<Integer,Integer> nodeClusterMap = new HashMap<Integer,Integer>();	
	HashMap<pair,Boolean> ClustQueryMap = new HashMap<pair,Boolean>();	
	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	Map<Integer, ArrayList<String>> inverted_list1 = new HashMap<Integer, ArrayList<String>>();
	Map<Integer, ArrayList<String>> inverted_list2 = new HashMap<Integer, ArrayList<String>>();
	Instances TrainingSet;
	HashMap<String,Boolean> dict = new HashMap<String,Boolean>();

	HashMap<Integer,String> recordList1 = new HashMap<Integer, String>();
	HashMap<Integer,String> recordList2 = new HashMap<Integer, String>();
	
	HashMap<String,Double> blockWeight1 = new HashMap<String, Double>();
	HashMap<String,Double> blockWeight2 = new HashMap<String, Double>();

	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();

	ArrayList<Integer> NewNodes = new ArrayList<Integer>();
	HashMap<String,Integer> numGreen = new HashMap<String, Integer>();
	HashMap<String,Integer> numRed = new HashMap<String, Integer>();

	HashMap<String,Integer> blockIndex1 = new HashMap<String, Integer>();
	HashMap<String,Integer> blockIndex2 = new HashMap<String, Integer>();

	HashMap<Integer,String> Index2Str1 = new HashMap<Integer, String>();
	HashMap<Integer,String> Index2Str2 = new HashMap<Integer, String>();

	ArrayList<ArrayList<Integer>> blockList1 = new ArrayList<ArrayList<Integer>>();
	ArrayList<ArrayList<Integer>> blockList2 = new ArrayList<ArrayList<Integer>>();

	ArrayList<double[]> blockSize1 = new ArrayList<double[]>();
	ArrayList<double[]> blockSize2 = new ArrayList<double[]>();

	HashMap<String,Integer> ruleBlockSize = new HashMap<String,Integer>();
	HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

	
	HashMap<pair,Integer> goldMap = new HashMap<pair,Integer>();

	HashMap<String, ArrayList<String>> blockjoin = new HashMap<String,ArrayList<String>>();
	ArrayList<String> blockRem = new ArrayList<String>();
	//ArrayList<HashMap<Integer,Double>> recordAdjacencyList = new ArrayList<HashMap<Integer,Double>>();

	HashMap<Integer, Double> expectedSize = new HashMap<Integer, Double>();
	String[] featnames= {"title","authors","year","journal","volume","pages","misc","sim"};

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
	public double Overlap (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0;
		double length1 = 0,length2=0;
		for(String a:s1.keySet()){
			if(s2.containsKey(a)){
				inter+=Math.min(s2.get(a), s1.get(a));
			}
			length1+=s1.get(a);
		}
		for(String a2:s2.keySet()) {
			length2 +=s2.get(a2);
		}
		return inter*1.0/Math.min(length1, length2);
	}
	public double CosineSim (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double num=0,den1=0,den2=0;
		for(String a:s1.keySet()){
			if(s2.containsKey(a)){
				num+=(s2.get(a)*s1.get(a));
				den2+=(s2.get(a)*s2.get(a));
			}
			den1+=(s1.get(a)*s1.get(a));
		}
		for(String a:s2.keySet()){
			if(s1.containsKey(a))
				continue;
			else
				den2+=(s2.get(a)*s2.get(a));

		}
		return num*1.0/(Math.sqrt(den1*den2));
	}

	//Overlapping tokens of high weight
	//Longest common subsequence?
	//

	//Weighted version of each feature

	//Longest common subsequence
	//Token overlap with high weights

	ArrayList<Integer> goodcount = new ArrayList<Integer>();
	ArrayList<Integer> badcount = new ArrayList<Integer>();
	public double Weightedjaccard (int u, int v){
		double inter = 0, union = 0;
		ArrayList<String> l1 =  inverted_list1.get(u);
		ArrayList<String> l2 =  inverted_list2.get(v);
		for(String a:l1){
			if(!blockWeight1.containsKey(a))
				continue;
	//		System.out.println(a+" "+u+" "+blockWeight1.get(a)+" "+union);
			if(l2.contains(a)){
				inter+=(blockWeight1.get(a));//*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight1.get(a));//*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight1.get(a));//*s1.get(a));
		}
		for(String a:l2){
			if(!blockWeight1.containsKey(a))
				continue;
			//System.out.println(a+" "+u+" "+blockWeight1.get(a)+" "+union);
			if(l1.contains(a))
				continue;
			else
				union+=(blockWeight1.get(a));//*s2.get(a));	
		}
		
		if (union==0.0)
			return 0.0;
			return inter*1.0/(union);
	}

	
	public HashMap<String,Integer> GetDict(String recordString) {
		recordString = recordString.toLowerCase( );
		String[] tokenList = recordString.split(" ",0);
		int val;
		HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
		//int recordId = Integer.parseInt(tokenList[0].trim());
		//if(!largest.contains(recordId))

		for (int i=1;i<tokenList.length;i++){
			if(tokenList[i].length()<=1)
				continue;
			if(tokenList[i].equals(";"))
				continue;
			
			Boolean found=false;
			String token=tokenList[i];
			if(blockRem.contains(token)) {
				continue;
			}
			if(recordMap.containsKey(token)){
				val = recordMap.get(token)+1;
				recordMap.put(token,val);
				continue;//Avoiding double insertion of same element in the block
			}
			recordMap.put(token,1);
			
			//if(token.contains("mk2"))
			//	System.out.println("dodge:"+blockjoin.get("mk2"));
			while(true) {
				if(blockjoin.containsKey(token)) {
					int iter=0;
					String initial=token;
					for(String s:blockjoin.get(token)) {
						
						if(recordString.contains(" "+s+" ")) {
							token = s+"!"+token;
							found=true;
							if(recordMap.containsKey(token)){
								val = recordMap.get(token)+1;
								recordMap.put(token,val);
								continue;//Avoiding double insertion of same element in the block
							}
							recordMap.put(token,1);
							break;
							
						}	
						iter++;
					}
					//System.out.println(token+" "+blockjoin.get(token)+" "+blockjoin.containsKey(token));
					if(iter == blockjoin.get(initial).size() )
						break;
					
				}
				else break;
			}
			
			
			/*if(recordMap.containsKey(token)){
				val = recordMap.get(token)+1;
				recordMap.put(token,val);
				continue;//Avoiding double insertion of same element in the block
			}
			recordMap.put(token,1);*/
			//ArrayList<Integer> blockContent;
		}
		return recordMap;
	}

	public ArrayList<double[]> get_benefit_component(int u){
		ArrayList<double[]> comp_lst = new ArrayList<double[]>();


		int iter = 0;
		while(iter<set_clusters.size()){
			component tmp = set_clusters.get(iter);
			ArrayList<Integer> nodes = tmp.get_component();
			int origsize=nodes.size();
			if(nodes.size()>100) {
				Collections.shuffle(nodes);
               			 nodes=new ArrayList<Integer>(nodes.subList(0, 100));
			}
			double prob = 0;
			for(int n : nodes){
				pair ed = new pair(u,n);
				if(edge_prob.containsKey(ed)) 
					prob+=edge_prob.get(ed);
			}
			double[] entry = {prob*nodes.size()*1.0/Math.min(100, nodes.size()),iter};


			comp_lst.add(entry);
			iter++;
		}


		Collections.sort(comp_lst, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[0]; double s2 = o2[0];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});
		return comp_lst;
	}
	public boolean[] query_edge_prob(HashMap<pair,Boolean>  oracle, HashMap<pair, Integer> gt, int u, int v, double r_to_g, double g_to_r) throws Exception{
		boolean ret[] = {false, false};
		pair tmp = new pair(u,v);
		pair tmp1 = new pair(v,u);
		if(gt.containsKey(tmp)||gt.containsKey(tmp1)) {
			ret[0]=true;
		}
		
		/*if(useClassifier) {
			if (Test(tmp,gt.get(u).equals(gt.get(v)))>0.5)
				ret[1]=true;
			else
				ret[1]=false;

		}else*/
			ret[1]=ret[0];	
			
			
		/*if(oracle.get(tmp))
			ret[1]=true;
		else
			ret[1]=false;
		 */
		
		//ret[1]=false;
		//Ground truth
		
		//ret[1]=ret[0];
		return ret;
	}

	/*public boolean[] query_edge_prob( HashMap<Integer, Integer> gt, int u, int v, double r_to_g, double g_to_r){
		boolean[] ret = {false,false};
		if(gt.get(u).equals(gt.get(v))){
			ret[0]=true;
			ret[1]=true;
		}

		return ret;
	}
	*/
	
	public void UpdateGreen(int n1, int n2){
		HashMap<String,Integer> s1 = GetDict(recordList1.get(n1));
		HashMap<String,Integer> s2 = GetDict(recordList2.get(n2));
		for(String s:s1.keySet()){
			if(s2.containsKey(s)){
				int val = 0;
				if(numGreen.containsKey(s))
					val = numGreen.get(s);
				numGreen.put(s, val+1);
				ArrayList<Integer> lst;
				if(blockQueries.containsKey(s))
				 lst = blockQueries.get(s);
				else
					lst = new ArrayList<Integer>();
	
				if (!lst.contains(n1))
					lst.add(n1);
				if (!lst.contains(n2))
					lst.add(n2);
				blockQueries.put(s, lst);
			}	
		}
	}
	public void UpdateRed(int n1, int n2){
		HashMap<String,Integer> s1 = GetDict(recordList1.get(n1));
		HashMap<String,Integer> s2 = GetDict(recordList2.get(n2));
		for(String s:s1.keySet()){
			if(s2.containsKey(s)){
				int val = 0;
				if(numRed.containsKey(s))
					val = numRed.get(s);
				numRed.put(s, val+1);
				
				ArrayList<Integer> lst;
				if(blockQueries.containsKey(s))
				 lst = blockQueries.get(s);
				else
					lst = new ArrayList<Integer>();
				//ArrayList<Integer> lst = blockQueries.get(s);
				if (!lst.contains(n1))
					lst.add(n1);
				if (!lst.contains(n2))
					lst.add(n2);
				blockQueries.put(s, lst);
			}	
		}
		
	}
	public void UpdateInferredEdges(){

		//numRed.clear();
		System.out.println(NewNodes.size()+"new nodes here" );
		for(int a:NewNodes){
			HashMap<String,Integer> s1 = GetDict(recordList1.get(a));
			/*boolean needfeedback=false;
			for(String s:s1.keySet()){
				if(blockQueries.containsKey(s)){
					int level=s.split("!",0).length - 1;
					System.out.println(s+" "+level+"level updateinferred");
					if(blockQueries.get(s).size()>100 || BlockTreeMap.get(level).get(s).size() == blockQueries.get(s).size())
                        			continue;
				}
				else
					needfeedback=true;
			}
			if(!needfeedback)
				continue;
			*/
			component c = set_clusters.get(nodeClusterMap.get(a));
			for(int b:c.get_component()){
				if (a==b)
					continue;
				pair t1 = new pair(a,b);
                                pair t2 = new pair(b,a);
                                //queried_edge_map.put(t1, true);
                                //queried_edge_map.put(t2, true);
                                UpdateGreen(a,b);
			}


			 for(int c1=0;c1<set_clusters.size();c1++){
				if(c1==nodeClusterMap.get(a))
					continue;
				pair p=new pair(c1,nodeClusterMap.get(a));
				if (ClustQueryMap.containsKey(p)){
					for(int b:set_clusters.get(c1).get_component())
						UpdateRed(a,b);
				}

			}
		}

		/*
		for(int c1=0;c1<set_clusters.size();c1++){
			//Update green and red counter between the two
			component c = set_clusters.get(c1);
			ArrayList<Integer> curr  = c.get_component();
			/*for(int i=0;i<curr.size();i++){
				for(int j=i+1;j<curr.size();j++){
					pair t1 = new pair(curr.get(i),curr.get(j));
					pair t2 = new pair(curr.get(i),curr.get(j));
					queried_edge_map.put(t1, true);
					queried_edge_map.put(t2, true);

					UpdateGreen(curr.get(i),curr.get(j));//Update green  between these two records

				}
			}
			
			for(int c2=c1+1;c2<set_clusters.size();c2++){
				component second = set_clusters.get(c2);
				ArrayList<Integer> secondcurr  = second.get_component();
				boolean found = false;
				for(int u:curr){
					for(int v:secondcurr){
						pair t = new pair(u,v);
						if(queried_edge_map.containsKey(t)){
							found = true;
							break;
						}
					}
					if(found)
						break;
				}
				if(found){
					for(int u:curr){
						for(int v:secondcurr){
							pair t1 = new pair(u,v);
							pair t2 = new pair(v,u);
							queried_edge_map.put(t1, false);
							queried_edge_map.put(t2, false);
							UpdateRed(u,v);
						}
					}
				}
			}
		}*/

	}
	
	public boolean UpdateWeight() throws FileNotFoundException{
		boolean updated = false;
		
		PrintStream bw = new PrintStream("bw.txt");
		PrintStream bwand = new PrintStream("and.txt");
		for (int i=0;i<BlockTreeMap1.keySet().size();i++) {
			HashMap<String, ArrayList<Integer>> level = BlockTreeMap1.get(i);
			int j=0;
			for(String s:level.keySet()) {
		
				double ng=0, nr=0;
				if(numGreen.containsKey(s))
					ng = numGreen.get(s);
				if(numRed.containsKey(s))
					nr = numRed.get(s);
				int queried = (int) (ng+nr);
				//int newqueried=queried;
				int newqueried=0;
				if(blockQueries.containsKey(s))
					newqueried =blockQueries.get(s).size();
				
				if (PrevQueries.containsKey(s)) {
						newqueried -=   PrevQueries.get(s);
						if(PrevQueries.get(s)>100)
							continue;
				}
				
				if(newqueried<10  )
					continue;
				if(blockQueries.containsKey(s))
				PrevQueries.put(s, blockQueries.get(s).size());
				
				
				ArrayList<Integer> samplelst = blockQueries.get(s);
				ArrayList<Integer> lstcopy  = level.get(s);
				
				int sizeneeded = 100-samplelst.size();
				if (sizeneeded < 0)
					sizeneeded = 0;
				updated=true;
				ArrayList<Integer> lst = new ArrayList<Integer>();
				for(int a:lstcopy) {
					if (!samplelst.contains(a))
							lst.add(a);
				}
				



				
				
				Collections.shuffle(lst);
				lst=new ArrayList<Integer>(lst.subList(0, Math.min(sizeneeded,lst.size())));
				lst.addAll(samplelst);



				/*		
				ArrayList<Integer> lst  = level.get(s);
				Collections.shuffle(lst);
				lst=new ArrayList<Integer>(lst.subList(0, Math.min(100,lst.size())));
				//System.out.println("block is "+s+" "+lst.size()+" "+i);
				
				j++;

				double ng=0, nr=0;
				if(numGreen.containsKey(s))
					ng = numGreen.get(s);
				if(numRed.containsKey(s))
					nr = numRed.get(s);
				int queried = (int) (ng+nr);
				*/

					//if(ng+nr> blockList.get(blockIndex.get(s)).size()){
				//if(nr+ng>0)
				HashMap<Integer,Integer> clustsize = new HashMap<Integer,Integer>();
				{
					//Get the cluster distribution for the block
					//Get the edges between clusters 

					double gr=0,rd=0;
					HashMap<Integer,Double> clustSize = new HashMap<Integer,Double>();
					for(int i1 = 0;i1<lst.size();i1++){
						int clusid = goldMap.get(lst.get(i1));
						//if(s.equals("drake'sjollyrodger2010imperialredale"))
						//	System.out.println(clusid+" "+lst.get(i1)+"this beer");
						if(clustsize.containsKey(clusid))
							clustsize.put(clusid, clustsize.get(clusid)+1);
						else
							clustsize.put(clusid, 1);
						if(!clustSize.containsKey(lst.get(i1)))
							clustSize.put(lst.get(i1), 0.0);
						for(int i2=i1+1;i2<lst.size();i2++){
							pair t=  new pair(lst.get(i1), lst.get(i2));
							if(goldMap.get(lst.get(i1)).equals(goldMap.get(lst.get(i2))))
								gr++;
							else
								rd++;

						//	if(nodeClusterMap.containsKey(t.x) and nodeClusterMap.containsKey(t.y)){
                                                  //      	if(nodeClusterMap.get(t.x)==nodeClusterMap.get(t.y)){
							/*if(queried_edge_map.containsKey(t)){
								//System.out.println("problem ehre");
								if(queried_edge_map.get(t)) {
									if(clustSize.containsKey(lst.get(i1)))
										clustSize.put(lst.get(i1),clustSize.get(lst.get(i1))+1);
									else
										clustSize.put(lst.get(i1),1.0);
									if(clustSize.containsKey(lst.get(i2)))
										clustSize.put(lst.get(i2),clustSize.get(lst.get(i2))+1);
									else
										clustSize.put(lst.get(i2),1.0);
								}

								continue;
							}
							*/if(nodeClusterMap.containsKey(t.x) && nodeClusterMap.containsKey(t.y)){
                                                        if(nodeClusterMap.get(t.x)==nodeClusterMap.get(t.y)){
                                                                if(clustSize.containsKey(lst.get(i1)))
                                                                        clustSize.put(lst.get(i1),clustSize.get(lst.get(i1))+1);
                                                                else
                                                                        clustSize.put(lst.get(i1),1.0);
                                                                if(clustSize.containsKey(lst.get(i2)))
                                                                        clustSize.put(lst.get(i2),clustSize.get(lst.get(i2))+1);
                                                                else
                                                                        clustSize.put(lst.get(i2),1.0);
                                                        }else{
                                                                pair cpair=new pair(nodeClusterMap.get(t.x),nodeClusterMap.get(t.y));
                                                                if (ClustQueryMap.containsKey(cpair))
                                                                        continue;
                                                                else{
                                                                        double p = 0.0;
                                                                        if(edge_prob.containsKey(t))
                                                                                p = edge_prob.get(t);

                                                                        if(clustSize.containsKey(lst.get(i1)))
                                                                                clustSize.put(lst.get(i1),clustSize.get(lst.get(i1))+p);
                                                                        else
                                                                                clustSize.put(lst.get(i1),1.0);
                                                                        if(clustSize.containsKey(lst.get(i2)))
                                                                                clustSize.put(lst.get(i2),clustSize.get(lst.get(i2))+p);
                                                                        else
                                                                                clustSize.put(lst.get(i2),1.0);

                                                                        ng+=p;
                                                                }
                                                        }
                                                        //ClustQueryMap.put(p1,false);
                                                        }
							else{
								double p = 0.0;
								if(edge_prob.containsKey(t))
									p = edge_prob.get(t);

								if(clustSize.containsKey(lst.get(i1)))
									clustSize.put(lst.get(i1),clustSize.get(lst.get(i1))+p);
								else
									clustSize.put(lst.get(i1),1.0);
								if(clustSize.containsKey(lst.get(i2)))
									clustSize.put(lst.get(i2),clustSize.get(lst.get(i2))+p);
								else
									clustSize.put(lst.get(i2),1.0);

								ng+=p;
								nr+=(1.0-p);

							}
						}
					}

					Map<Integer, Double> sortedMap = 
							clustSize.entrySet().stream()
							.sorted(Entry.comparingByValue(Comparator.reverseOrder()))
							.collect(Collectors.toMap(Entry::getKey, Entry::getValue,
									(e1, e2) -> e1, LinkedHashMap::new));


					gr +=lst.size();//n choose2 +n
					ng +=lst.size();//n choose2 +n
					if(nr+ng>0){
						//System.out.println(gr*1.0/(gr+rd));
						int maxsize = 0;
						double dummysize = 0;
						double entropy = 0.0;
						double clustDist  = 0.0;
						for(int a:clustsize.keySet()) {
							if(clustsize.get(a)>maxsize)
								maxsize = clustsize.get(a);
							entropy += (clustsize.get(a)*1.0/lst.size()) *Math.log(clustsize.get(a)*1.0/lst.size());
							clustDist += (clustsize.get(a)*1.0/lst.size());
						}
						//entropy=Math.abs(entropy);
						int numclust = 0;
						int currclust = 0;
						for(int a: sortedMap.keySet()) {
							if(sortedMap.get(a)>dummysize)
								dummysize = clustSize.get(a);
							if(currclust==0) {
								currclust = (int) Math.ceil(sortedMap.get(a));
								numclust++;
							}
							else
								currclust--;
						}

						//bw.println(s+" "+lst.size()+" "+Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())+" "+gr*1.0/(gr+rd)+" "+(ng*1.0/(ng+nr))+" "+maxsize*1.0/lst.size()+" "+1.0/clustsize.keySet().size()+" "+(gr*1.0/(gr+rd))*(1.0/clustsize.keySet().size())*(maxsize*1.0/lst.size()));//+" "+maxsize+" "+clustsize.keySet().size()+" "+maxsize*1.0/lst.size());//(gr*1.0/(gr+rd)));
						//bw.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd)+" "+ng*1.0/(ng+nr)+" "+(gr*1.0/(gr+rd))*1.0/(ng*1.0/(ng+nr)));
						//blockWeight.put(blockIndex.get(s), (ng*1.0/(ng+nr)));

						double weighted=0.0, normal=0.0;

						//if(gr*1.0/(gr+rd) >= 0.99 && lst.size()>1) 
						{
							for(int i11=0;i11<lst.size();i11++) {
								int u = lst.get(i11);
								for(int j1=i11+1;j1<lst.size();j1++) {

									int v = lst.get(j1);
									weighted += Weightedjaccard(u,v);
									normal += jaccard(GetDict(recordList1.get(u)),GetDict(recordList2.get(v)));
								}
							}
							weighted = weighted*2.0/(lst.size()*(lst.size()-1));
							normal = normal*2.0/(lst.size()*(lst.size()-1));
						//	System.out.println("block is "+s+" "+lst.size()+" "+weighted+" "+normal+" "+i);
						}

						
						//blockWeight.put(blockIndex.get(s), Math.pow((ng*1.0/(ng+nr)),1)*(maxsize*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())
						int fulldummy = 1;
						//1 = full dummy
						//0 = Full converged
						//2 = Only number of clusters is not converged
						//3 = entropy is not converged
						//4 = only fraction of green edges is not converged
						//5 = Use only converged fraction of green edges
						//6 = Entropy
						//System.out.println(entropy);
						if(fulldummy ==1)
							blockWeight1.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if(fulldummy == 0)
							blockWeight1.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy== 2)
							blockWeight1.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy==3) {
							if(queried>10)
								blockWeight1.put(s, Math.pow((gr*1.0/(gr+rd)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
							else
								blockWeight1.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						}else if (fulldummy ==4)
							blockWeight1.put(s, Math.pow(ng*1.0/(ng+nr),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy==5)
							blockWeight1.put(s, Math.pow(gr*1.0/(gr+rd),1));
						else if (fulldummy==6) {
							//if(lst.size()>1)
							blockWeight1.put(s, (gr*1.0/(gr+rd))*Math.exp(entropy) );//(maxsize*1.0/lst.size())//clustsize.keySet().size()		
						}
						//bw.println(i+" "+s+" "+lst.size()+" "+gr*1.0/(gr+rd)+" "+weighted+" "+normal+" "+Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())+" "+blockWeight.get(blockIndex.get(s)));


						//if(lst.size()>200)
						//blockWeight.put(blockIndex.get(s), (gr*1.0/(gr+rd)));
						//else
						//	blockWeight.put(blockIndex.get(s), Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size()));

					}
				}
				
			}
		}
		
		return updated;
	}
	
		
	public void GetExpected() throws FileNotFoundException{
		//System.out.println(queried_edge_map.keySet().size()+"size is ");
		SimilarityList.clear();
		//edge_prob.clear();
		expectedSize.clear();		
		//USE processed_nodes here and partition similarity in buckets so it is easy to store vslues for each bucket
		//
		PrintStream pt = new PrintStream("blockweight.txt");

	
		HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
		Comparator<String> comparator = new ValueComparator(blockWeight1);
		
		TreeMap<String, Double> result = new TreeMap<String, Double>(comparator);
		result.putAll(blockWeight1);

		int iter=0;
		for (String key : result.keySet()) { 

			String s = key;
			int level = s.split("!",0).length-1;
			ArrayList<Integer> lst1 = BlockTreeMap1.get(level).get(s);
			ArrayList<Integer> lst2 = BlockTreeMap2.get(level).get(s);
			//System.out.println(iter+" "+key+" "+lst.size()+" "+blockWeight.get(key)+" "+processed.keySet().size());
			if(iter%10000==0)
			//if(level>0)
				System.out.println(iter+" "+s+" "+lst1.size());//+"k"+BlockTreeMap1.get(level-1).containsKey(s));//+" "+lst1.size());//+" "+lst1.size()+" "+blockWeight1.get(s));
			iter++;
			if (iter >= 0.992*result.keySet().size()){
				if(s.equals("here") || s.equals("without"))
					System.out.println("found here "+s);
				continue;	
				

			}
			//if (lst.size()>3000 && !s.equals("chevrolet"))
			//	continue;
			
			//double gr=0,rd=0;
			for(int i1=0;i1<lst1.size();i1++){
				int u = lst1.get(i1);
				for(int i2=i1+1;i2<lst2.size();i2++){
					int v = lst2.get(i2);
					//if(goldMap.get(u).equals(goldMap.get(v)))
					//	gr++;
					//else rd++;
					pair t = new pair(u,v);
					pair t1 = new pair(v,u);
					if(processed.containsKey(t) || processed.containsKey(t1))
						continue;
					else {
						double[] p = {Weightedjaccard(u,v), u, v};
						SimilarityList.add(p);
						
						processed.put(t1, true);
						processed.put(t, true);
					}
				}
			}

			//if(lst.size()>1)
			//	pt.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd));
		}
		pt.close();
		processed.clear();
		//System.out.println("similarity list size is "+SimilarityList.size());
		//System.out.println(queried_edge_map.keySet().size()+"size is ");



		//This is probabilit conversion of sim
		Collections.sort(SimilarityList, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[0]; double s2 = o2[0];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});

		edge_prob.clear();
		double maxsim = SimilarityList.get(0)[0];
		//System.out.println("maximum similarity is "+maxsim+" "+goldMap.get((int)SimilarityList.get(0)[1])+" "+goldMap.get((int)SimilarityList.get(0)[2]));
		ArrayList<double[]> bucket =new ArrayList<double[]>();
		double max = 1.0,min = 0.99;
		double gr =0;
		double rd = 0;
		double gtgr=0,gtrd=0;
		PrintStream probvales = new PrintStream("probvalues.txt");
		expectedSize.clear();
		double numEdges = 0,prevgr=0,prevrd=0;
		//System.out.println(queried_edge_map.keySet().size()+"size is ");

		for(int i=0 ;i<SimilarityList.size();i++){
			double[] curr = SimilarityList.get(i) ;
			//if (i<100)
			//System.out.println(curr[0]+" "+curr[1]+" "+curr[2]+" "+goldMap.get((int)curr[1])+" "+goldMap.get((int)curr[2]));
			//curr[0] = curr[0]*1.0/(maxsim);
			if(curr[0] > 1.0)
				curr[0]=1.0;
			//Amplify the similarity because all similarities are in 0-0.4
			if(curr[0]<min || i==SimilarityList.size()-1){
			//	if(min==0.009999999999999247)
			//		System.out.println("22DDDDDD"+gr+ " "+rd+" "+i+" "+queried_edge_map.keySet().size()+" "+bucket.size()+" "+min);
				
				
				if(i==SimilarityList.size()-1)
					bucket.add(curr);
				if(bucket.size()>100 || i==SimilarityList.size()-1 ){
					
					if(min==0.009999999999999247)
						System.out.println("DDDDDD"+gr+ " "+rd+" "+i+" "+queried_edge_map.keySet().size()+" "+bucket.size()+" "+min);
					
					for (double[] p:bucket){
						int u = (int) p[1];
						int v = (int) p[2];
						double pr =0;
						if(gr+rd>0)
							pr= gr*1.0/(gr+rd);
						else
							pr = p[0];

						if (gtgr+gtrd>100)
							pr=gtgr*1.0/(gtgr+gtrd);
						pair t1 = new pair(u,v);
						pair t2 = new pair(v,u);
						if(u==994 && (v==1812))
							System.out.println(p[0]+"fjsdkfhsdj "+pr+" "+gr+" "+rd+" "+bucket.size()+" "+numEdges+" "+queried_edge_map.keySet().size());
						edge_prob.put(t1, pr);
						edge_prob.put(t2, pr);
						double val1 = 0.0;
						if( expectedSize.containsKey(u))
							val1=expectedSize.get(u);
						double val2 = 0.0;
						if( expectedSize.containsKey(v))
							val2=expectedSize.get(v);

						val1+=pr;
						val2+=pr;
						expectedSize.put(u, val1);
						expectedSize.put(v, val2);
			//			probvales.println(u+" "+v+" "+p[0]+" "+pr+" "+goldMap.get(u).equals(goldMap.get(v)));
					}
					if(numEdges>100) {
						prevgr=gr;
						prevrd=rd;
					}
					gr=0;
					rd=0;
					gtgr=0;
					gtrd=0;
					numEdges=0;

					//for(double[] p:bucket)
					//probprint.println(p[0]+" "+gr*1.0/(gr+rd));
					//probprint.println(bucket.get(0)[0]+" "+bucket.get(bucket.size()-1)[0]+" "+gr*1.0/(gr+rd)+" "+bucket.size());
					bucket.clear();
					max=min;
					min = max-0.01;
					//bucket.add(curr);
					//continue;
				}else{
					min-=0.01;
				}
			}
			bucket.add(curr);
			pair t = new pair((int)SimilarityList.get(i)[1],(int)SimilarityList.get(i)[2]);
			
			{
				//System.out.println((int)SimilarityList.get(i)[1]+" "+(int)SimilarityList.get(i)[2]);
				double p=0;
				if( !pipelinetype.equals("ploop")){
					if(nodeClusterMap.containsKey(t.x) && nodeClusterMap.containsKey(t.y)){
						if(nodeClusterMap.get(t.x)==nodeClusterMap.get(t.y))
							p=1.0;
						else{
						
							pair cpair=new pair(nodeClusterMap.get(t.x),nodeClusterMap.get(t.y));
							if(ClustQueryMap.containsKey(cpair))
								p=0.0;
							else{
								//if(goldMap.get(t.x).equals(goldMap.get(t.y)))
								//	p=1.0;
								//else
								//	p=0;
								p=curr[0];
							}
						}
					}else{
						p=curr[0];
						//f(goldMap.get(t.x).equals(goldMap.get(t.y)))
						//	p=1.0;
						//else
						//	p=0;
					}

				//	if(queried_edge_map.get(t))
				//		p=1.0;
				//	else
				//		p=0;
					numEdges++;
				}else {
				//	if(goldMap.get(t.x).equals(goldMap.get(t.y)))
				//		p=1.0;
				//	else
				//		p=0;
					p= curr[0];
				}
				//if(u==994 && (v==1812))
				//if(curr[1]==994 && curr[2]==1812)
				gr+=p;
				rd+=(1.0-p);
				if((p==0.0) || (p==1.0)){
					if (goldMap.get(t.x).equals(goldMap.get(t.y)))
						gtgr+=1.0;
					else
						gtrd+=1.0;

				}
				//if(min==0.009999999999999247)
				//	System.out.println(p+"DDDDDD"+gr+ " "+rd+" "+i+" "+queried_edge_map.keySet().size()+" "+bucket.size()+" "+min);
				
			}


		}
		//SimilarityList.clear();	
	}

	public ArrayList<double[]> get_benefit_clusters(){
		ArrayList<double[]> clust_benefit = new ArrayList<double[]>();
		for(int i=0;i<set_clusters.size();i++) {
			ArrayList<Integer> clus1 = set_clusters.get(i).get_component();
			for(int j=i+1;j<set_clusters.size();j++) {
				ArrayList<Integer> clus2 = set_clusters.get(j).get_component();
				boolean found = false;
				double benefit=0.0;
				if(nodeClusterMap.containsKey(clus1.get(0)) && nodeClusterMap.containsKey(clus2.get(0))){
					if (nodeClusterMap.get(clus1.get(0)) == nodeClusterMap.get(clus2.get(0)) )
						found=true;

				}
				if(!found)
				for(int a:clus1) {
					for(int b:clus2) {
						pair t = new pair(a,b);
						if(queried_edge_map.containsKey(t)) {
							found = true;
							break;
						}
						if(edge_prob.containsKey(t))
							benefit+=edge_prob.get(t);
						else {
							//System.out.println(a+" "+b+" "+i+" "+j+" "+clus1.size()+" "+clus2.size());

							//System.out.println("not present"+edge_prob.keySet().size());
						}
					}
					if(found)
						break;
				}
				if(!found) {
					double[] tmp = {benefit,i,j};
					clust_benefit.add(tmp);
				}
			}
		}

		Collections.sort(clust_benefit, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[0]; double s2 = o2[0];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});
		return clust_benefit;
	}
	
	public ArrayList<String> Intersect(String a, String b) {
		ArrayList<String> c = new ArrayList<String>();
		//System.out.println(a+" "+b);
		HashMap<String,Integer> bdict = GetDict(b);
		HashMap<String,Integer> adict = GetDict(a);
		for(String akey: adict.keySet()) {
			if(bdict.containsKey(akey)) {
				c.add(akey);
			}
		}
		return c;
	}
	public ArrayList<HashMap<String,Double[]>> getRules (ClassifierTree t) throws Exception {
		ArrayList<HashMap<String,Double[]>> lst = new ArrayList<HashMap<String,Double[]>>();
		if(t.isLeaf()) {
			HashMap<String,Double[]> ruleMap = new HashMap<String,Double[]>();
			String color = t.getLocalModel().dumpLabel(0, t.getTrainingData()).split(" ",0)[0];
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
			
			ArrayList<HashMap<String,Double[]>>  o1 =  getRules(t.getSons()[0]);
			
			ArrayList<HashMap<String,Double[]>>  o2 =  getRules(t.getSons()[1]);
			
			System.out.println(t.getLocalModel().rightSide(0,t.getTrainingData()).split(" ",0)[2]);
			Double[] l = {-1.0,  Double.parseDouble(t.getLocalModel().rightSide(0,t.getTrainingData()).split(" ",0)[2])};
			for(HashMap<String,Double[]> map : o1) {
				map.put(t.getLocalModel().leftSide(t.getTrainingData())+";"+Math.random(),l);
				lst.add(map);
			}
			
			System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2]);
			Double[] r = {1.0,  Double.parseDouble(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2])};
			for(HashMap<String,Double[]> map : o2) {
				map.put(t.getLocalModel().leftSide(t.getTrainingData())+";"+Math.random(),r);
				lst.add(map);
			}
			//ProcessTree(t.getSons()[1]);
			return lst;
		}
	
		//return ruleMap;
	}
	/*public void getRules(ClassifierTree t) {
		ArrayList<HashMap<String,Double[]>> classifier = new ArrayList<HashMap<String,Double[]>>();
		//HashMap<String,Double[]> ruleMap = new HashMap<String,Double[]>();
		 
		
		
		while(true) {
		 if (t.isLeaf()) {
			 System.out.println(t.getLocalModel().dumpLabel(0, t.getTrainingData()));
			break; 
		 }else {
			 System.out.println(t.getLocalModel().leftSide(t.getTrainingData()));
			 System.out.println(t.getLocalModel().rightSide(0,t.getTrainingData()));
			 System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()));

			 t=t.getSons()[0];
		 }
		}
	}*/
	
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
				map.put(featnames[t.m_Attribute]+";"+Math.random(),l);
				lst.add(map);
			}
			
			//System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2]);
			Double[] r = {1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o2) {
				map.put(featnames[t.m_Attribute]+";"+Math.random(),r);
				lst.add(map);
			}
			
			return lst;
		}
	
		//return ruleMap;
	} 
	
	
	public boolean satisfyRule(HashMap<String,Double[]> rule, HashMap<String,Double> feat) {
		// boolean holds=true;
		for(String s:rule.keySet()) {
			 if(s.trim().equals("class"))
				 continue;
			 String s1 = s.split(";",0)[0];
			 if(feat.get(s1.trim())  <= rule.get(s)[1] && rule.get(s)[0]<0 )
				 continue;
			 else if (feat.get(s1.trim())  > rule.get(s)[1] && rule.get(s)[0]>0)
				 continue;
			 else {
				 return false;
			 }
			 //System.out.print(s+" "+r.get(s)[0]+" "+r.get(s)[1]+" ;;");
		 }
		return true;
	}
	public ArrayList<Integer> GetIntersection(ArrayList<Integer> lst1, ArrayList<Integer> lst2){
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		for (int a:lst1) {
			if(lst2.contains(a))
				tmp.add(a);
		}
		return tmp;
	}
	
	public int mergeBlocks() {
		//POpulate ruleBlockSize
		
		for(int jj=1;jj<=2;jj++) {
			Map<Integer, ArrayList<String>> inverted_list = new HashMap<Integer, ArrayList<String>>();
			int depth=2;
			HashMap<Integer,HashMap<String,ArrayList<Integer>>> tmpBlockTreeMap = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();
			
			for (int i=0;i<depth;i++) {
				HashMap<String,ArrayList<Integer>> tmp = new HashMap<String,ArrayList<Integer>>();
				tmpBlockTreeMap.put(i,tmp);
			}
			
			HashMap<Integer,String> recordList;
			if(jj==1)
				recordList= recordList1;
			else
				recordList= recordList2;
			
			
			for(Integer rec_id : recordList.keySet()) {
				String[] rec = recordList.get(rec_id).split(" ",0);
				int i=0;
				HashMap<String,Boolean> alreadyDone=new HashMap<String,Boolean>();
				for(;i<rec.length;i++) {
					int j=i+1;
					while (j<rec.length) {
						String token = String.join("!", Arrays.copyOfRange(rec, i, j));
						if (alreadyDone.containsKey(token)) {
							j++;
							continue;
						}else {
							alreadyDone.put(token, true);
						}

						int level = j-i-1;
						if (level>=depth) {
							j++;
							continue;
						}
						//System.out.println(i+" "+j+" "+token+" "+level);
						HashMap<String,ArrayList<Integer>> level_map = tmpBlockTreeMap.get(level);
			
						if(token.equals("(instrumental)!"))
							System.out.println(rec+" "+recordList.get(rec_id));
						if(level_map.containsKey(token)) {
							ArrayList<Integer> lst = level_map.get(token);
							lst.add(rec_id);
							level_map.put(token, lst);
						}else {
							ArrayList<Integer> lst = new ArrayList<Integer>();
							lst.add(rec_id);
							level_map.put(token, lst);
						}
						
						tmpBlockTreeMap.put(level, level_map);
						
						j++;
					}
				}
				//System.out.println("thos"+BlockTree.get(1).keySet().size());//+" "+level+" "+level_map.keySet().size());
			}
			//System.out.println("thos"+tmpBlockTree.get(1).keySet());//+" "+level+" "+level_map.keySet().size());
			for(int i=0;i<recordList.size();i++){
				ArrayList<String> tmp_inverted_lst=new ArrayList<String>();
				inverted_list.put(i, tmp_inverted_lst);
			}

			//System.out.println("thos"+BlockTree.get(1).keySet().size());//+" "+level+" "+level_map.keySet().size());
			
			System.out.println("done reging");
			//Pruning here
			
			int counter=0;
			int loc=0;
			for(int i=0;i<depth;i++) {
				HashMap<String,ArrayList<Integer>> level_map =tmpBlockTreeMap.get(i);
				HashMap<String,ArrayList<Integer>> cleaned_level_map = new HashMap<String,ArrayList<Integer>>();
				//System.out.println("thos"+BlockTree.get(1).keySet().size());//+" "+level+" "+level_map.keySet().size());
				//System.out.println(i+" "+level_map.keySet().size());
				
				for(String s:level_map.keySet()) {

					if(s.equals("(instrumental)!"))
						System.out.println(s+" "+level_map.get(s).size()+" "+i);
					String[] tokenLst = s.split("!");
					//if(level_map.get(s).size()>10 || tokenLst.length==1)
					{
						//If it is correlated then make this else no
						

						if (tokenLst.length>1) {
							String parent1= String.join("!", Arrays.copyOfRange(tokenLst, 0, tokenLst.length-1));
							String parent2= String.join("!", Arrays.copyOfRange(tokenLst, 1, tokenLst.length));
							//System.out.println(parent1+" "+parent2+" "+tokenLst.length);
							double p1 = tmpBlockTreeMap.get(tokenLst.length-2).get(parent1).size()*1.0/recordList.size();
							double p2 = tmpBlockTreeMap.get(tokenLst.length-2).get(parent2).size()*1.0/recordList.size();
							double curr_p = level_map.get(s).size()*1.0/recordList.size();
							if (curr_p > p1*p2 ) {
								cleaned_level_map.put(s, level_map.get(s));
								for(int a:level_map.get(s)) {
									ArrayList<String> tmp = inverted_list.get(a);
									tmp.add(s);
									inverted_list.put(a, tmp);
								}
								counter++;
							}
						}else {
							cleaned_level_map.put(s, level_map.get(s));
							for(int a:level_map.get(s)) {
								
								ArrayList<String> tmp = inverted_list.get(a);
								//System.out.println(a+" "+s);
								tmp.add(s);
								inverted_list.put(a, tmp);
							}
							counter++;
							if(jj==1) {
								blockIndex1.put(s,loc);
								Index2Str1.put(loc, s);
								double[] tmp = {loc,level_map.get(s).size()};
								blockSize1.add(tmp);
							}else {
								blockIndex2.put(s,loc);
								Index2Str2.put(loc, s);
								double[] tmp = {loc,level_map.get(s).size()};
								blockSize2.add(tmp);
							}
							
							loc+=1;
						}
					}
				}
				if(jj==1) {
					inverted_list1=inverted_list;
					BlockTreeMap1.put(i, cleaned_level_map);
				}
				else {
					inverted_list2=inverted_list;
					BlockTreeMap2.put(i, cleaned_level_map);
				}
			}
			
			
			Collections.sort(blockSize1, new Comparator<double[]>() {
				public int compare(double[] o1, double[] o2) {
					double s1 = o1[1]; double s2 = o2[1];
					if (s1 != s2)
						return (s1 > s2 ? 1 : -1);
					else
						return 0;
				}
			});
			Collections.sort(blockSize2, new Comparator<double[]>() {
				public int compare(double[] o1, double[] o2) {
					double s1 = o1[1]; double s2 = o2[1];
					if (s1 != s2)
						return (s1 > s2 ? 1 : -1);
					else
						return 0;
				}
			});
			
			//How do we weigh a pair of records
			System.out.println("done reging"+counter);
		}
		
		return 0	;
	}
	
	public void processBlocks() throws Exception{
		if (mergeBlocks()==0) {
			System.out.println("done merging");
			//return;
		}
			
		
		
		recallprint= new PrintStream(folder+"/recall.txt");
	//	PrintStream Nodeorder= new PrintStream("nodeorder.txt");
		
//		HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

		int blockId=0;
		
		float totalpairs=(float) (Math.log(recordList1.keySet().size())+ Math.log(recordList2.keySet().size()));//recordList1.keySet().size()*recordList2.keySet().size();
		blockWeight1.clear();
		for(int i=0;i<BlockTreeMap1.keySet().size();i++) {
			HashMap<String,ArrayList<Integer>> level = BlockTreeMap1.get(i);
			for(String s:level.keySet()) {
				//if (i==0) 
				{
					if(!BlockTreeMap2.get(i).containsKey(s))
						continue;
					if (BlockTreeMap2.get(i).get(s).size()==0){
						System.out.println("problem "+s);
						return;
					}
					double score1 = recordList1.keySet().size()*1.0/level.get(s).size();
					double score2 = recordList2.keySet().size()*1.0/BlockTreeMap2.get(i).get(s).size();
					//Nodeorder.println(s+" "+level.get(s).size()+" "+BlockTreeMap2.get(i).get(s).size());
					if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
						blockWeight1.put(s, Math.log(score1*score2)*1.0/(totalpairs));
					}else {
						blockWeight1.put(s, 1.0);
					}	
				}
				//else {
				//	blockWeight1.put(s, 0.0);
				
					
		/*		if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
					blockWeight.put(s, Math.log(recordList.keySet().size()*1.0/level.get(s).size())*1.0/Math.log(recordList.keySet().size()));
				}else {
					blockWeight.put(s, 1.0);
				}*/
		//	}	
		}
		}
		
		
		if(blockWeight1.keySet().size()>0) {
			System.out.println("Number of blocks "+blockWeight1.size());
		}
		
		
		
		
		//Calculate expected cluster size
		//Update the block weights
		//Store an inverted list i.e. blocks for each record so we can quickly update weights
			//It will help in calculating the weighted jaccard
			
		//CLASSIFIER
		//We need to learn a classifier and use crowd if it is not confident
		//Variation1: No classifier
		//Variation2: Just classifier and crowd to train the classifier
		//Variation3: Classifier to guide the queries for the crowd
		
		System.gc();
		System.out.println("process1 KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		
		//numIgnore = blockSize.size()/4;
		PrintStream simlist = new PrintStream(folder+"/sim.txt");
		//Assumption in code we have id form 0 to n-1
		//This can be reduced but for later by iterating over blocks and not n choose 2
		int i1=0;
		for( int i=0;i<1;i++){
			i1=1;
			System.out.println(i);
			GetExpected();
			System.out.println("done expected");	
		//	if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop"))
		//		boolean updated=UpdateWeight();	
			System.out.println("done updateweight");	
		}
		
		
		
		/*for(int i=0;i<1000;i++) {
			double[] curr = SimilarityList.get(i) ;
			boolean out =false;
			pair p1 = new pair((int)curr[1],(int)curr[2]);
			pair p2 = new pair((int)curr[2],(int)curr[1]);
			System.out.println(p1.x+" "+p2.x+" "+goldMap.containsKey(p1));
			if( goldMap.containsKey(p1) || goldMap.containsKey(p2)) {
				
				if(goldMap.get(p1)>0)
					out=true;
			}
			System.out.println(curr[0]+" "+curr[1]+" "+curr[2]+" "+out);
		}
		if(i1>0)
			return;
		*/
		
		//if(pipelinetype.equals("pM3loop")|| pipelinetype.equals("ploop"))
		//	GetExpected();
		//if(i1>0)
		//	return;
		System.gc();
		System.out.println("process2 KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		
		System.out.println("done expected");	

		
		
		System.out.println(edge_prob.keySet().size()+"edgeprob"+" "+SimilarityList.size());
		
		int curr_iter = 0;
		
		
		HashMap<Integer,Integer> clust_map = new HashMap<Integer,Integer>();
		//HashMap<Integer,ArrayList<Integer>> inverseclust_map = new HashMap<Integer,ArrayList<Integer>>();
		
		int gr=0;
		while(true){
			

			boolean end=false;
			for(int i=curr_iter;i<SimilarityList.size();) {
				double[] curr = SimilarityList.get(i);
				int u = (int)curr[1];
				int v= (int) curr[2];
				if(u==v)
					continue;
				
				boolean[] output =  query_edge_prob( oracle, goldMap, u, v,0,0);
				if(output[1]) {
					if(clust_map.containsKey(u) && clust_map.containsKey(v)) {
						if(!clust_map.get(u).equals(clust_map.get(v))) {
							ArrayList<Integer> c1 = set_clusters.get(clust_map.get(u)).get_component();
							ArrayList<Integer> c2 = set_clusters.get(clust_map.get(v)).get_component();
							gr+= c1.size()*c2.size();
							int old=clust_map.get(v);
							for(int a:c2) {
								clust_map.put(a, clust_map.get(u));
							}
							c1.addAll(c2);
							component t = new component (c1);
							set_clusters.set(clust_map.get(u), t);
							
							ArrayList<Integer> tp = new ArrayList<Integer>();
							component cc = new component(tp);
							set_clusters.set(old,cc);
							
							
							//System.out.println(gr+" jumped by "+ c1.size()*c2.size());
						}
					}else {
						
						if(clust_map.containsKey(u)) {
							ArrayList<Integer> tmp = set_clusters.get(clust_map.get(u)).get_component();
							tmp.add(v);
							//System.out.println(tmp+" "+v+" "+u+" "+clust_map.containsKey(u)+" "+clust_map.containsKey(v));
							component c = new component(tmp);
							set_clusters.set(clust_map.get(u),c);
							clust_map.put(v, clust_map.get(u));
							//inverseclust_map.put(set_clusters.size()-1, tmp);
							gr+= tmp.size()-1 ;
						}else if(clust_map.containsKey(v)){
							ArrayList<Integer> tmp = set_clusters.get(clust_map.get(v)).get_component();
							tmp.add(u);
							component c = new component(tmp);
							set_clusters.set(clust_map.get(v),c);
							clust_map.put(u, clust_map.get(v));
							//inverseclust_map.put(set_clusters.size()-1, tmp);
							gr+= tmp.size()-1 ;
						}
						else {
							ArrayList<Integer> tmp = new ArrayList<Integer>();
							tmp.add(u);
							tmp.add(v);
							component c = new component(tmp);
							set_clusters.add(c);
							clust_map.put(u, set_clusters.size()-1);
							clust_map.put(v, set_clusters.size()-1);
							//inverseclust_map.put(set_clusters.size()-1, tmp);
							gr++;
						}
					}
				}
				else{
					
					;//System.out.println(u+" "+v+" "+curr[0]);
					//System.out.println(recordList1.get(u));
					//System.out.println(recordList1.get(v));

				}				
				
				i++;
				if(i%100 == 0) {
					curr_iter=i;
					//Send feedback
					break;
				}
				
				if(i==SimilarityList.size())
					end=true;
			}
			recallprint.println(curr_iter+" "+gr*2.0/goldMap.keySet().size()+" "+gr);	
			System.out.println(gr+" "+curr_iter+" "+goldMap.keySet().size());	
			if (end)
				break;
		}
		int iter=0;
		for(pair p : goldMap.keySet()) {
			if(!edge_prob.containsKey(p)) {
				iter++;
				if(clust_map.containsKey(p.x) &&  clust_map.containsKey(p.y)) {
					if(clust_map.get(p.x).equals(clust_map.get(p.y)))
						System.out.println(p.x+" "+p.y);
				}
			}
		}
		System.out.println(iter+" "+edge_prob.keySet().size());
		this.recallprint.println(processed_nodes.size()+" "+this.N);
		/*gr=0;
		for(int i=0;i<set_clusters.size();i++) {
			ArrayList<Integer> c = set_clusters.get(i).get_component();
			gr+= ((c.size()*(c.size()-1))*0.5);
			if(c.size()>2)
				System.out.println(c);
			
		}
		
		System.out.println(gr);
		*/
	}
	

	
	public static void main(String[] args) throws Exception{

		CleanWithoutClassifierBlocking FF = new CleanWithoutClassifierBlocking();
		System.out.println("Starting Memory KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
		String[] attributeList ;
		String line;
		int val;
		for(int i=1;i<=2;i++) {
			String recordFile     = FF.folder+"/records"+Integer.toString(i)+".txt";
			Scanner scanner = new Scanner(new File(recordFile));
			
			HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
			
			while(scanner.hasNextLine()){
				line= scanner.nextLine();
				line = line.toLowerCase( );
				line=line.replace("!", "");
				line = line.replaceAll("( )+", " ");

				//System.out.println(line);
				attributeList = line.split(" ",0);
				//ArrayList<String> record = new ArrayList<String>();
				HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
				int recordId = Integer.parseInt(attributeList[0].trim().replace(".jpg", ""))-1;
				//if(FF.folder=="cora")
				//	recordId++;
				//if(!largest.contains(recordId))
					String[] tokenList = attributeList;
					for(int j=1;j<tokenList.length;j++) {
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
				if(i==1)
					FF.recordList1.put(recordId, line);
				else
					FF.recordList2.put(recordId, line);
			}
			

			tmpblockList.clear();
			tmpblockList=null;
			System.gc();
		}
		
		
		
		//Free blockList and blockSize and form the lists....
		int loc=0;
		System.out.println("Memoery after forming blocks: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		

		/*for (String s:tmpblockList.keySet()){	
			FF.blockIndex.put(s,loc);
			FF.Index2Str.put(loc, s);
			double[] tmp = {loc,tmpblockList.get(s).size()};
			FF.blockSize.add(tmp);
			loc+=1;
		}*/

		System.out.println("Done with all block variable setting: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);

	

		

		System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);


		String goldFile     = FF.folder+"/gold.txt";

		Scanner goldScanner = new Scanner(new File(goldFile));
		while(goldScanner.hasNextLine()){
			line= goldScanner.nextLine();
			String[] goldtokens = line.split(" ",0);
			int u = Integer.parseInt(goldtokens[0].replace(".jpg", ""))-1;
			int id = Integer.parseInt(goldtokens[1])-1;
		//	if(u>100000 || id > 100000)
		//		continue;
			pair p1 = new pair(u,id);
			FF.goldMap.put(p1, 1);
		}
			//FF.goldMap.put(p2, 1);
		
		
		
		System.out.println("Memory after goldMap:" + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		System.out.println("Going inside process");
		FF.processBlocks();

	}

}

