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


public class BlossCleanClassifierBlockingsongs {
	Random rand = new Random(1991);
	static Random generator = new Random(1992);

	String folder = "songsClassifier";//"citations";//"songs";
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
	boolean useClassifier=false;//true;
	int mergedBlocks=-1;
	boolean classifierExists=false;
	FastVector fvWekaAttributes;
	int featNumbers= 9;
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap1 = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap2 = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();

	int num_blocks=0;
	
	HashMap<String, ArrayList<Integer>> blockQueries = new HashMap<String,ArrayList<Integer>>();
	HashMap<String, Integer> PrevQueries = new HashMap<String,Integer>();

	Set<Integer> processed_nodes = new HashSet<Integer>();
	HashMap<Integer,Integer> nodeClusterMap = new HashMap<Integer,Integer>();	
	HashMap<pair,Boolean> ClustQueryMap = new HashMap<pair,Boolean>();	
	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> inferred_edges = new HashMap<pair, Boolean>();
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
	ArrayList<String> featnames= new ArrayList<String>();//{"title","authors","year","journal","volume","pages","misc","sim"};

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
		ArrayList<String> l2 =  inverted_list1.get(v);
		for(String a:l1){
			
		//	System.out.println(a+" "+u+" "+v+" "+union+" "+blockWeight1.containsKey("a")+" "+l2);
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
				Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
				Collections.shuffle(nodes,newgen);
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
		
		if(useClassifier) {
			if(inferred_edges.containsKey(tmp) || inferred_edges.containsKey(tmp1))
				ret[1]=true;
			/*double classifierout = 0.0;
				if (gt.containsKey(tmp))
					classifierout = Test(tmp,true,ModelRuleslst);
				else
					classifierout = Test(tmp,false,ModelRuleslst);
			if (classifierout > 0.5)
				ret[1]=true;
			else
				ret[1]=false;
			*/
		}else
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
				



				
				Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));	
				Collections.shuffle(lst,newgen);
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
		HashMap<Integer,Integer> node_degree1 = new HashMap<Integer,Integer>();
                HashMap<Integer,Integer> node_degree2 = new HashMap<Integer,Integer>();


	
		HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
		Comparator<String> comparator = new ValueComparator(blockWeight1);
		
		TreeMap<String, Double> result = new TreeMap<String, Double>(comparator);
		result.putAll(blockWeight1);

		int iter=0;
		for (String key : result.keySet()) { 

			String s = key;
			int level = s.split("!",0).length-1;
			ArrayList<Integer> lst1 = BlockTreeMap1.get(level).get(s);
			if(lst1.size()==1)
				continue;
			ArrayList<Integer> lst2 = BlockTreeMap1.get(level).get(s);
			//System.out.println(iter+" "+key+" "+lst.size()+" "+blockWeight.get(key)+" "+processed.keySet().size());
			if(iter%10000==0)
			//if(level>0)
				System.out.println(iter+" "+s+" "+lst1.size()+" "+SimilarityList.size());//+"k"+BlockTreeMap1.get(level-1).containsKey(s));//+" "+lst1.size());//+" "+lst1.size()+" "+blockWeight1.get(s));
			iter++;


			if (SimilarityList.size()>10000000)
				continue;
			/*
			if (iter >= 0.98*result.keySet().size()){
				if(s.equals("here") || s.equals("without"))
					System.out.println("found here "+s);
				continue;	
				

			}*/
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
						edge_prob.put(t, p[0]);
                                                edge_prob.put(t1, p[0]);
						if(node_degree1.containsKey(u))
                                                        node_degree1.put(u,node_degree1.get(u)+1);
                                                else
                                                        node_degree1.put(u,1);
                                                if(node_degree2.containsKey(v))
                                                        node_degree2.put(v,node_degree2.get(v)+1);
                                                else
                                                        node_degree2.put(v,1);				
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

		//edge_prob.clear();
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
/*
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
*/		//SimilarityList.clear();	

		ArrayList<ArrayList<pair>> levelList = new ArrayList<ArrayList<pair>>();
                HashMap<pair,ArrayList<Integer>> blocking_features = new HashMap<pair,ArrayList<Integer>>();
                for(int aa=0;aa<500;aa++){
                        ArrayList<pair> tmp = new ArrayList<pair>();
                        levelList.add(tmp);
                }
                double blossT=50;
                HashMap<Integer,Integer> blocking_edges = new HashMap<Integer,Integer>();
                for(int i=0;i<SimilarityList.size();i++){
                        double[] curr = SimilarityList.get(i);
                        pair p = new pair((int)curr[1],(int)curr[2]);
                         if(blocking_edges.containsKey(p.x))
                                        blocking_edges.put(p.x,blocking_edges.get(p.x)+1);
                                else
                                        blocking_edges.put(p.x,1);
                                if(blocking_edges.containsKey(p.y))
                                        blocking_edges.put(p.y,blocking_edges.get(p.y)+1);
                                else
                                        blocking_edges.put(p.y,1);
                                if(blocking_edges.get(p.x)>100 || blocking_edges.get(p.y)>100){
                                        ;
                                        continue;
                                                    }


                        double[] cfibf = get_cficf_score(p);
                        int loc = (int)(cfibf[2]*1.0/blossT);
                        ArrayList<pair> tmp = levelList.get(loc);
                        ArrayList<Integer> feattmp = new ArrayList<Integer>();
                        feattmp.add(Math.min(100,node_degree1.get(p.x)));
                        feattmp.add(Math.min(100,node_degree2.get(p.y)));
                        feattmp.add((int)cfibf[0]);
                        feattmp.add((int)cfibf[1]*100);
                        feattmp.add((int)cfibf[2]*100);
                        blocking_features.put(p,feattmp);
                        tmp.add(p);
                        levelList.set(loc,tmp);

                }
                System.out.println(blocking_features.keySet().size());
                process_pairs_levels(levelList,blocking_features);






	}
	 public pair get_best_pair(HashMap<String,Boolean> covered_features, ArrayList<pair> level, HashMap<pair,ArrayList<Integer>> blocking_features){
                ArrayList<int[]> feat_benefit = new ArrayList<int[]>();
                for (pair p : level){
                        int score =0;
                        ArrayList<Integer> feat = blocking_features.get(p);
                        if(!covered_features.containsKey("a"+feat.get(0).toString()))
                                score++;
                        if(!covered_features.containsKey("b"+feat.get(1).toString()))
                                score++;
                        if(!covered_features.containsKey("c"+feat.get(2).toString()))
                                score++;
                        if(!covered_features.containsKey("d"+feat.get(3).toString()))
                                score++;
                        if(!covered_features.containsKey("e"+feat.get(4).toString()))
                                score++;
                        int[] tmp = {score,p.x,p.y};
                        if(score>1)
                                feat_benefit.add(tmp);

                }
                if(feat_benefit.size()==0)
                        return null;
                Collections.sort(feat_benefit, new Comparator<int[]>() {
                        public int compare(int[] o1, int[] o2) {
                                int s1 = o1[0]; int s2 = o2[0];
                                if (s1 != s2)
                                        return (s1 > s2 ? -1 : 1);
                                else
                                        return 0;
                        }
                });
                pair ret= new pair(feat_benefit.get(0)[1],feat_benefit.get(0)[2]);
                return ret;
        }
	 public void process_pairs_levels(ArrayList<ArrayList<pair>> levelList, HashMap<pair,ArrayList<Integer>> blocking_features) throws FileNotFoundException{//,Exception{
                ArrayList<pair> training = new ArrayList<pair>();
		//select random pairs here
		//
		int numpairperlevel=50;
		ArrayList<pair> final_candidates = new ArrayList<pair>();
		for (ArrayList<pair> level: levelList){
                        Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
			Collections.shuffle(level,newgen);
			final_candidates.addAll(level.subList(0, Math.min(50,level.size())));

		}

                //for (ArrayList<pair> level: levelList)
                {
			Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
                        Collections.shuffle(final_candidates,newgen);
                        HashMap<String,Boolean> covered_features = new HashMap<String,Boolean>();
                        while(true){
                                pair best = get_best_pair(covered_features,final_candidates,blocking_features);
                                if(best==null)
                                        break;
                                else{
                                        training.add(best);
                                         ArrayList<Integer> feat = blocking_features.get(best);
                                         covered_features.put("a"+feat.get(0).toString(),true);
                                         covered_features.put("b"+feat.get(1).toString(),true);
                                         covered_features.put("c"+feat.get(2).toString(),true);
                                         covered_features.put("d"+feat.get(3).toString(),true);
                                         covered_features.put("e"+feat.get(4).toString(),true);

                                }
                                System.out.println(training.size()+" "+final_candidates.size()+" "+covered_features.keySet().size());
                        }
                }
                int gr=0,rd=0;
                double prob = 0.0;

                for(pair p:training){
                        if(goldMap.containsKey(p))
                                gr++;
                        else{
                                rd++;
                                prob+=edge_prob.get(p);
                        }

                }
                System.out.println(gr+" "+rd+" "+prob*1.0/rd);
                double avg = prob*1.0/rd;
                int higher = 0;
		PrintStream pt = new PrintStream(folder+"/Blosstrain.txt");
                PrintStream blossfalcon = new PrintStream(folder+"/Blossfalcon.txt");
                int it=0;
                for(pair p : training){
                        ArrayList<Integer> feat = blocking_features.get(p);
                        if(it==0) pt.println("a,b,c,d,e,f,id1,id2,Class");
                        for (int f:feat){
                                pt.print(f+",");
                        }
                        if(goldMap.containsKey(p))
                                pt.print(edge_prob.get(p)+","+p.x+","+p.y+",1.0\n");
                        else    pt.print(edge_prob.get(p)+","+p.x+","+p.y+",0.0\n");

                        HashMap<String,Double> falconfeat = new HashMap<String,Double>();

                        if(goldMap.containsKey(p))
                        falconfeat=Extractfeat(p,true);
                        else
                        falconfeat=Extractfeat(p,false);
                        if (it==0){
                          for (String s: falconfeat.keySet()){
                                  if(!s.equals("Class"))
                                          blossfalcon.print(s+",");
                          }

                          blossfalcon.print("Class\n");
                                it+=1;
                        }



                        for (String s: falconfeat.keySet()){
                                    if( s.equals("Class"))
                                                                                                                                                                                         
                                continue;
                                                if(falconfeat.get(s).isNaN())
                                                        blossfalcon.print("0.0,");
                                                else blossfalcon.print(falconfeat.get(s)+",");

                           }
                                                blossfalcon.print(falconfeat.get("Class")+"\n");
                }

                pt.close();
                blossfalcon.close();
                PrintStream btest = new PrintStream(folder+"/Blosstest.txt");
                it=0;
                for(pair p : blocking_features.keySet()){
                        if(training.contains(p))
                                continue;

                        if (it==0){
                                btest.println("a,b,c,d,e,f,id1,id2,Class");
                                it+=1;
                        }
                        ArrayList<Integer> feat = blocking_features.get(p);
                        for (int f:feat){
                                btest.print(f+",");
                        }
                        if(goldMap.containsKey(p))
                                btest.print(edge_prob.get(p)+","+p.x+","+p.y+",1.0\n");
                        else    btest.print(edge_prob.get(p)+","+p.x+","+p.y+",0.0\n");
                }
                btest.close();


                String falcon_data = folder+"/falcondata.txt";
                Scanner scanner = new Scanner(new File(falcon_data));

                PrintStream falcon_output = new PrintStream(folder+"/falcon_train.txt");
                String line;
                it=0;
                 while(scanner.hasNextLine()){
                       line= scanner.nextLine();
                        String[] idlist = line.split(",",0);
                        HashMap<String,Double> feat = new HashMap<String,Double>();
                        pair p = new pair(Integer.parseInt(idlist[0])-1,Integer.parseInt(idlist[1])-1);
                        if(idlist[2].equals("t"))
                                feat=Extractfeat(p,true);
                        else
                                feat=Extractfeat(p,false);
                        if (it==0){
                          for (String s: feat.keySet()){
                                  if(!s.equals("Class"))
                                          falcon_output.print(s+",");
                          }
                          falcon_output.print("Class\n");
                                it+=1;
                        }
                             for (String s: feat.keySet()){
                                    if( s.equals("Class"))
                                                        continue;
                                                if(feat.get(s).isNaN())
                                                        falcon_output.print("0.0,");
                                                else falcon_output.print(feat.get(s)+",");

                           }
                                                falcon_output.print(feat.get("Class")+"\n");
                }
                falcon_output.close();
	    String[] args1 = new String[] {"/bin/bash", "-c", "cd ~/javaCode/src/songsClassifier; python learnerbloss.py > b1; cd .."};
                try{
                Process proc = new ProcessBuilder(args1).start();
                proc.waitFor();
                }catch(Exception e){
                        ;
                }
                String blossoutput = folder+"/blossoutput.txt";
                scanner = new Scanner(new File(blossoutput));
                SimilarityList.clear();
                while(scanner.hasNextLine()){
                        line= scanner.nextLine();
                        String[] idlist = line.split(" ",0);

                        double[] entry = {Double.valueOf(idlist[2]),Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1])};
                        SimilarityList.add(entry);
                }
                 Collections.sort(SimilarityList, new Comparator<double[]>() {
                        public int compare(double[] o1, double[] o2) {
                                double s1 = o1[0]; double s2 = o2[0];
                                if (s1 != s2)
                                        return (s1 > s2 ? -1 : 1);
                                else
                                        return 0;
                        }
                });
        }

	 public double[] get_cficf_score(pair p){
                double score=0.0,reciprocal_score=0;
                int inter=0;
                ArrayList<String> list2 = inverted_list1.get(p.y);
		int num1=0;
                for (String s1: inverted_list1.get(p.x)){
                                int level = s1.split("!",0).length-1;
                                if(level>0)
                                        continue;
				num1++;
		 if(list2.contains(s1)){
                                inter++;
                                reciprocal_score+=1.0/(BlockTreeMap1.get(level).get(s1).size() + BlockTreeMap1.get(level).get(s1).size());
                        }
                }

		int num2=0;
		for (String s1: inverted_list1.get(p.y)){
                                int level = s1.split("!",0).length-1;
                                if(level>0)
                                        continue;
                                num2++;
		}
                double[] ret_score = {inter,reciprocal_score, inter*Math.log(num_blocks*1.0/num1)*Math.log(num_blocks*1.0/num2)};
                return ret_score;

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
				map.put(featnames.get(t.m_Attribute)+";"+Math.random(),l);
				lst.add(map);
			}
			
			//System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2]);
			Double[] r = {1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o2) {
				map.put(featnames.get(t.m_Attribute)+";"+Math.random(),r);
				lst.add(map);
			}
			
			return lst;
		}
	
		//return ruleMap;
	} 
	
	
	
	public ArrayList<Integer> GetIntersection(ArrayList<Integer> lst1, ArrayList<Integer> lst2){
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		for (int a:lst1) {
			if(lst2.contains(a))
				tmp.add(a);
		}
		return tmp;
	}
        public  ArrayList<String> get_parent_list(String s,  HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap){
		ArrayList<String> parent_lst = new ArrayList<String>();

		String[] tokenLst = s.split("!",0);
		if(tokenLst.length==1){
			return parent_lst;
		}
		String parent1;
		parent1= String.join("!", Arrays.copyOfRange(tokenLst, 0, tokenLst.length-1));
		String parent2= String.join("!", Arrays.copyOfRange(tokenLst, 1, tokenLst.length));
		if(BlockTreeMap.get(tokenLst.length-2).containsKey(parent1))
			parent_lst.add(parent1);
		else
			parent_lst.addAll(get_parent_list(parent1,BlockTreeMap));
		if(BlockTreeMap.get(tokenLst.length-2).containsKey(parent2))
			parent_lst.add(parent2);
		else
			parent_lst.addAll(get_parent_list(parent2,BlockTreeMap));
		return parent_lst;
	}


                                                       
	public int mergeBlocks() {
		//POpulate ruleBlockSize
		
		for(int jj=1;jj<=1;jj++) {
			System.out.println("came back?");
			Map<Integer, ArrayList<String>> inverted_list = new HashMap<Integer, ArrayList<String>>();
			int depth=10;
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
				String tmp = recordList.get(rec_id);
				tmp=tmp.replace(";"," ");
				tmp = tmp.replaceAll("( )+", " ");
				
				String[] rec = tmp.split(" ",0);
				int i=0;
				HashMap<String,Boolean> alreadyDone=new HashMap<String,Boolean>();
				for(;i<rec.length;i++) {
					int j=i+1;
					while (j<=rec.length) {
						String token = String.join("!", Arrays.copyOfRange(rec, i, j));
						if (rec_id==959057 || rec_id== 154063)
							System.out.println(token+" "+i+" "+j);
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

					if(s.equals("0.698452538946!0.502974636418!1979"))
						System.out.println(s+" "+level_map.get(s).size()+" "+i+" "+level_map.get(s));
					String[] tokenLst = s.split("!");
					//if(level_map.get(s).size()>10 || tokenLst.length==1)
					{
						//If it is correlated then make this else no
						
						
						if( tokenLst.length>1 && level_map.get(s).size()==1)
							continue;	
						if (tokenLst.length>1) {
							ArrayList<String> parent_lst = new ArrayList<String>();
							if(jj==1) parent_lst = get_parent_list(s,BlockTreeMap1);
							if(jj==2) parent_lst = get_parent_list(s,BlockTreeMap2);
							double parent_p=1.0;
							for (String parent: parent_lst){
								int parent_level = parent.split("!",0).length-1;
								double p1 =tmpBlockTreeMap.get(parent_level).get(parent).size()*1.0/recordList.size(); 
								parent_p *= p1;
							}

							/*
							String parent1;

							parent1= String.join("!", Arrays.copyOfRange(tokenLst, 0, tokenLst.length-1));
							String parent2= String.join("!", Arrays.copyOfRange(tokenLst, 1, tokenLst.length));
							//System.out.println(parent1+" "+parent2+" "+tokenLst.length);
							double p1 = tmpBlockTreeMap.get(tokenLst.length-2).get(parent1).size()*1.0/recordList.size();
							double p2 = tmpBlockTreeMap.get(tokenLst.length-2).get(parent2).size()*1.0/recordList.size();
							double curr_p = level_map.get(s).size()*1.0/recordList.size();
							*/
							double curr_p = level_map.get(s).size()*1.0/recordList.size();
							if (curr_p > parent_p ) {
								cleaned_level_map.put(s, level_map.get(s));
								for(int a:level_map.get(s)) {
									ArrayList<String> tmp = inverted_list.get(a);
									tmp.add(s);
									inverted_list.put(a, tmp);
								}
								//num_blocks++;
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
							num_blocks++;
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
			tmpBlockTreeMap.clear();	
			//How do we weigh a pair of records
			System.out.println("done reging"+counter);
		}
		System.out.println("exiting this");
		return 0	;
	}
	
	public HashMap<String,Double> extractFeatures(pair p){
		HashMap<String,Double> featureVect = new HashMap<String,Double>();
		String rec1 = recordList1.get(p.x);
		String rec2 = recordList2.get(p.y);
		HashMap<String,Integer> recmap1 = GetDict(rec1);
		HashMap<String,Integer> recmap2 = GetDict(rec2);


		featureVect.put("jaccard", jaccard(recmap1,recmap2));
		featureVect.put("Overlap", Overlap(recmap1,recmap2));
		featureVect.put("weightedJaccard", Weightedjaccard(p.x,p.y));
		featureVect.put("Cosine", CosineSim(recmap1,recmap2));
		return featureVect;
	}
	
	public double Weightedjaccardold (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		
		for(String a:s1.keySet()){
			//System.out.println(a+"here "+blockWeight.containsKey(a));
			if(!blockWeight1.containsKey(a)) {
				System.out.println("found one"+a);
				continue;
			}
			if(s2.containsKey(a)){
				//System.out.println(blockWeight.keySet());
				inter+=(blockWeight1.get(a)*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight1.get(a)*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight1.get(a)*s1.get(a));
		}
		for(String a:s2.keySet()){
			if(!blockWeight1.containsKey(a))
				continue;
			
			if(s1.containsKey(a))
				continue;
			else
				union+=(blockWeight1.get(a)*s2.get(a));	
		}

		return inter*1.0/union;
	}
	
	
	public double[] JaccardOverlapDice (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		double l1=0,l2=0;
		for(String a:s1.keySet()){
			//System.out.println(a+"here "+blockWeight.containsKey(a));
			l1+=s1.get(a);
			
			if(s2.containsKey(a)){
				//System.out.println(blockWeight.keySet());
				inter+=Math.min(s2.get(a), s1.get(a));
				union+=Math.max(s2.get(a), s1.get(a));
			}else
				union+=s1.get(a);
		}
		for(String a:s2.keySet()){
			l2+=s2.get(a);
			if(s1.containsKey(a))
				continue;
			else
				union+=(s2.get(a));	
		}
		double[] stats = new double[]{inter*1.0/union, inter*1.0/(Math.min(l1, l2)), 2.0*inter*1.0/(l1+l2)}; 
		return stats ;
	}
	public double Cosine(HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double num=0,d1=0,d2=0;
		for(String a:s1.keySet()){
			//System.out.println(a+"here "+blockWeight.containsKey(a));
			d1+=s1.get(a)*s1.get(a);
			
			if(s2.containsKey(a)){
				//System.out.println(blockWeight.keySet());
				num+=(s2.get(a)*s1.get(a));
			}
		}
		for(String a:s2.keySet()){
			d2+=s2.get(a)*s2.get(a);	
		}
		return num*1.0/Math.sqrt(d1*d2);
	}
	
	private  int indexOf(char character, char[] buffer, int fromIndex,
			int toIndex, boolean[] matched) {

		// compare char with range of characters to either side
		for (int j = Math.max(0, fromIndex), length = Math.min(toIndex, buffer.length); j < length; j++) {
			// check if found
			if (buffer[j] == character && !matched[j]) {
				return j;
			}
		}

		return -1;
	}
	
	public int[] getCommonCharacters(final char[] charsA,
			final char[] charsB, final int separation) {
		final int[] common = new int[Math.min(charsA.length, charsB.length)];
		final boolean[] matched = new boolean[charsB.length];

		// Iterate of string a and find all characters that occur in b within
		// the separation distance. Mark any matches found to avoid
		// duplicate matchings.
		int commonIndex = 0;
		for (int i = 0, length = charsA.length; i < length; i++) {
			final char character = charsA[i];
			final int index = indexOf(character, charsB, i - separation, i
					+ separation + 1, matched);
			if (index > -1) {
				common[commonIndex++] = character;
				matched[index] = true;
			}
		}

		if (commonIndex < common.length) {
			common[commonIndex] = -1;
		}

		// Both invocations will yield the same multi-set terminated by -1, so
		// they can be compared for transposition without making a copy.
		return common;
	}

	
	public double Jaro( String a,  String b) {

		if (a.isEmpty() && b.isEmpty()) {
			return 1.0f;
		}

		if (a.isEmpty() || b.isEmpty()) {
			return 0.0f;
		}

		// Intentional integer division to round down.
		final int halfLength = Math.max(0, Math.max(a.length(), b.length()) / 2 - 1);

		final char[] charsA = a.toCharArray();
		final char[] charsB = b.toCharArray();
		final int[] commonA = getCommonCharacters(charsA, charsB, halfLength);
		final int[] commonB = getCommonCharacters(charsB, charsA, halfLength);

		// commonA and commonB will always contain the same multi-set of
		// characters. Because getCommonCharacters has been optimized, commonA
		// and commonB are -1-padded. So in this loop we count transposition
		// and use commonCharacters to determine the length of the multi-set.
		float transpositions = 0;
		int commonCharacters = 0;
		for (int length = commonA.length; commonCharacters < length
				&& commonA[commonCharacters] > -1; commonCharacters++) {
			if (commonA[commonCharacters] != commonB[commonCharacters]) {
				transpositions++;
			}
		}

		if (commonCharacters == 0) {
			return 0.0f;
		}

		float aCommonRatio = commonCharacters / (float) a.length();
		float bCommonRatio = commonCharacters / (float) b.length();
		float transpositionRatio = (commonCharacters - transpositions / 2.0f)
				/ commonCharacters;

		return (aCommonRatio + bCommonRatio + transpositionRatio) / 3.0f;
	}

	public String commonPrefix(String a, String b) {
	    int minLength = Math.min(a.length(), b.length());
	    for (int i = 0; i < minLength; i++) {
	        if (a.charAt(i) != b.charAt(i)) {
	            return a.substring(0, i);
	        }
	    }
	    return a.substring(0, minLength);
	}
	
	//https://github.com/Simmetrics/simmetrics/blob/master/simmetrics-core/src/main/java/org/simmetrics/metrics/JaroWinkler.java
	public double JaroWinkler( String a,  String b) {
		 double jaroScore = Jaro(a, b);
		int prefixLength = Math.min(commonPrefix(a, b).length(), 4);

		return  (jaroScore + (prefixLength * 0.1f * (1.0f - jaroScore)));
	}
	public double Levenshtein_sim (String a,String b) {
		return 1.0f - (Levenshtein_distance(a, b) / (  Math.max(a.length(), b.length())));
	}
	public double Levenshtein_distance(String s, String t) {
		float insertDelete=(float) 1.0;
		float substitute=(float) 1.0;
		if (s.isEmpty())
			return t.length();
		if (t.isEmpty())
			return s.length();
		if (s.equals(t))
			return 0;

		final int tLength = t.length();
		final int sLength = s.length();

		float[] swap;
		float[] v0 = new float[tLength + 1];
		float[] v1 = new float[tLength + 1];

		// initialize v0 (the previous row of distances)
		// this row is A[0][i]: edit distance for an empty s
		// the distance is just the number of characters to delete from t
		for (int i = 0; i < v0.length; i++) {
			v0[i] = i * insertDelete;
		}

		for (int i = 0; i < sLength; i++) {

			// first element of v1 is A[i+1][0]
			// edit distance is delete (i+1) chars from s to match empty t
			v1[0] = (i + 1) * insertDelete;

			for (int j = 0; j < tLength; j++) {
				v1[j + 1] = Math.min(Math.min(v1[j] + insertDelete,
						v0[j + 1] + insertDelete),
						v0[j]
								+ (s.charAt(i) == t.charAt(j) ? 0.0f
										: substitute));
			}

			swap = v0;
			v0 = v1;
			v1 = swap;
		}

		// latest results was in v1 which was swapped with v0
		return v0[tLength];
	}
	public HashMap<String,Integer> three_gram(String s){
		HashMap<String,Integer> three_gram_map= new HashMap<String,Integer>();
		String[] token_lst = s.split(" ",0);
		
		for (int i=2;i<token_lst.length;i++) {
			String three_gram = token_lst[i-2]+" "+token_lst[i-1]+" "+token_lst[i];
			int val=0;
			if (three_gram_map.containsKey(three_gram))
				val = three_gram_map.get(three_gram);
			three_gram_map.put(three_gram, val+1);
		}
		return three_gram_map;
	}
	public HashMap<String,Integer> three_gram_single_word(String s){
		HashMap<String,Integer> three_gram_map= new HashMap<String,Integer>();
		//String[] token_lst = s.split(" ",0);
		
		for (int i=2;i<s.length();i++) {
			String three_gram = s.charAt(i-2)+" "+s.charAt(i-1)+" "+s.charAt(i);
			int val=0;
			if (three_gram_map.containsKey(three_gram))
				val = three_gram_map.get(three_gram);
			three_gram_map.put(three_gram, val+1);
		}
		return three_gram_map;
	}
	
	public double mongeElkan(List<String> a, List<String> b) {
		// calculates average( for s in a | max( for q in b | metric(s,q))
		double sum = 0.0f;

		for (String s : a) {
			double max = 0.0f;
			for (String q : b) {
				max = Math.max(max, Levenshtein_sim(s, q));
			}
			sum += max;
		}
		return sum / a.size();
	}
	
	public double MatchMismatchcompare(String a, int aIndex, String b, int bIndex,double f1,double f2) {
		return a.charAt(aIndex) == b.charAt(bIndex) ? f2
				: f1;
	}
	public double needlemanWunchdist( String s,  String t) {

		if ((s.equals(t) )) {
			return 0;
		}

		if (s.isEmpty()) {
			return   t.length();
		}
		if (t.isEmpty()) {
			return   s.length();
		}
		
		 int n = s.length();
		 int m = t.length();

		// We're only interested in the alignment penalty between s and t
		// and not their actual alignment. This means we don't have to backtrack
		// through the n-by-m matrix and can safe some space by reusing v0 for
		// row i-1.
		double[] v0 = new double[m + 1];
		double[] v1 = new double[m + 1];

		for (int j = 0; j <= m; j++) {
			v0[j] = j;
		}

		for (int i = 1; i <= n; i++) {
			v1[0] = i;

			for (int j = 1; j <= m; j++) {
				v1[j] = Math.min(Math.min(
						v0[j]   , 
						v1[j - 1]), 
						v0[j - 1] - MatchMismatchcompare(s, i - 1, t, j - 1,-1,0));
			}
			
			 double[] swap = v0; v0 = v1; v1 = swap;

		}
		
		// Because we swapped the results are in v0.
		return v0[m];
	}
	public double needlemanWunch(String a, String b) {

		if (a.isEmpty() && b.isEmpty()) {
			return 1.0f;
		}

		double maxDistance = Math.max(a.length(), b.length())
				* Math.max(0.0, -2);
		double minDistance = Math.max(a.length(), b.length())
				* Math.min(-1, -2);

		return (-needlemanWunchdist(a, b) - minDistance)
				/ (maxDistance - minDistance);

	}
	
	public double smithWatermanGotoh(final String s, final String t) {
		
		double[] v0 = new double[t.length()];
		double[] v1 = new double[t.length()];

		double max = v0[0] = Math.max(Math.max(0, -0.5f), MatchMismatchcompare(s, 0, t, 0,-2.0,1.0));

		for (int j = 1; j < v0.length; j++) {
			v0[j] = Math.max(Math.max(0, v0[j - 1] + -0.5f),
					MatchMismatchcompare(s, 0, t, j,-2.0,1.0));

			max = Math.max(max, v0[j]);
		}

		// Find max
		for (int i = 1; i < s.length(); i++) {
			v1[0] = Math.max(Math.max(0, v0[0] + -0.5f), MatchMismatchcompare(s, i, t, 0,-2.0,1.0));

			max = Math.max(max, v1[0]);

			for (int j = 1; j < v0.length; j++) {
				v1[j] = Math.max(Math.max(Math.max(0, v0[j] + -0.5f), v1[j - 1] + -0.5f),
						v0[j - 1] + MatchMismatchcompare(s, i, t, j,-2.0,1.0));

				max = Math.max(max, v1[j]);
			}

			for (int j = 0; j < v0.length; j++) {
				v0[j] = v1[j];
			}
		}

		
		float maxDistance = Math.min(s.length(), t.length());
		return max*1.0 / maxDistance;
	}
	private double smithWaterman(String a, String b) {
		final int n = a.length();
		final int m = b.length();

		final double[][] d = new double[n][m];

		// Initialize corner
		double max = d[0][0] = Math.max(0, MatchMismatchcompare(a, 0, b, 0,5.0f, -3.0f));

		// Initialize edge
		for (int i = 0; i < n; i++) {

			// Find most optimal deletion
			double maxGapCost = 0;
			for (int k = Math.max(1, i - Integer.MAX_VALUE); k < i; k++) {
				maxGapCost = Math.max(maxGapCost, d[i - k][0] + (-5-1*(i-k-i-1)));//gap.value(i - k, i));
			}

			d[i][0] = Math.max(Math.max(0, maxGapCost), MatchMismatchcompare(a, i, b, 0,5.0f, -3.0f));

			max = Math.max(max, d[i][0]);

		}

		// Initialize edge
		for (int j = 1; j < m; j++) {

			// Find most optimal insertion
			double maxGapCost = 0;
			for (int k = Math.max(1, j - Integer.MAX_VALUE); k < j; k++) {
				maxGapCost = Math.max(maxGapCost, d[0][j - k] + (-5-1*(j-k-j-1)));//gap.value(j - k, j));
			}

			d[0][j] = Math.max(Math.max(0, maxGapCost), MatchMismatchcompare(a, 0, b, j,5.0f, -3.0f));

			max = Math.max(max, d[0][j]);

		}

		// Build matrix
		for (int i = 1; i < n; i++) {

			for (int j = 1; j < m; j++) {

				double maxGapCost = 0;
				// Find most optimal deletion
				for (int k = Math.max(1, i - Integer.MAX_VALUE); k < i; k++) {
					maxGapCost = Math.max(maxGapCost,
							d[i - k][j] +  (-5-1*(i-k-i-1)));
				}
				// Find most optimal insertion
				for (int k = Math.max(1, j - Integer.MAX_VALUE); k < j; k++) {
					maxGapCost = Math.max(maxGapCost,
							d[i][j - k] +  (-5-1*(j-k-j-1)));
				}

				// Find most optimal of insertion, deletion and substitution
				d[i][j] = Math.max(Math.max(0, maxGapCost),
						d[i - 1][j - 1] + MatchMismatchcompare(a, i, b, j,5.0f, -3.0f));

				max = Math.max(max, d[i][j]);
			}

		}

		
		
		float maxDistance = Math.min(a.length(), b.length())
				* Math.max(5, -5);
		return max*1.0/ maxDistance;
	}

	public HashMap<String,Double> Get_small_string_features(String a,String b){
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		
		HashMap<String,Integer> dict1 = GetDict("1 "+a);
		HashMap<String,Integer> dict2 = GetDict("2 "+b);
		//featMap.put("title", Weightedjaccardold(GetDict("1 "+a),GetDict("2 "+b)));
		
		double[] jacc = JaccardOverlapDice(three_gram(a),three_gram(b));
		featMap.put("jacc3", jacc[0]);
		//featMap.put("overlap3", jacc[1]);
		//featMap.put("dice3", jacc[2]);
		
		double[] jacc_word = JaccardOverlapDice(dict1,dict2);
		//featMap.put("jacc", jacc_word[0]);
		featMap.put("title_overlap", jacc_word[1]);
		//featMap.put("title_dice", jacc_word[2]);
		
		//featMap.put("cosine", Cosine(dict1,dict2));
		/*featMap.put("monge_elkan", mongeElkan(Arrays.asList(a.split(" ",0)),Arrays.asList(b.split(" ",0))));
		featMap.put("needleman",needlemanWunch(a,b));
		featMap.put("smithWaterman",smithWaterman(a,b));
		featMap.put("smithWatermangotoh",smithWatermanGotoh(a,b));
		*/
		return featMap;
	}
	public HashMap<String,Double>  Get_Single_Word_features(String a, String b){
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		//if(a.equals(b))
		//	featMap.put("exact_match", 1.0);
		//else
		//	featMap.put("exact_match", 0.0);
		
		double[] jacc = JaccardOverlapDice(three_gram_single_word(a),three_gram_single_word(b));
		featMap.put("jacc3", jacc[0]);
		//featMap.put("overlap3", jacc[1]);
		//featMap.put("dice3", jacc[2]);
		
		//featMap.put("levenshtein", Levenshtein_sim (a,b) );
		//featMap.put("jaro", Jaro (a,b) );
		//featMap.put("jarowinkler", JaroWinkler (a,b) );
		return featMap;
	}
	public HashMap<String,Double>    Get_Numeric_features(String a,String b){
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		double v1 = Double.valueOf(a);
		double v2 = Double.valueOf(b);
		if(v1==v2)
			featMap.put("exact_match", 1.0);
		else
			featMap.put("exact_match", 0.0);
		
		//featMap.put("absolute_diff", Math.abs(v1-v2));
		//featMap.put("relative_diff", Math.abs((v1-v2)*1.0/v1));
		//featMap.put("levenshtein", Levenshtein_sim (a,b) );
		
		return featMap;
	}
	public HashMap<String,Double> Extractfeat(pair p, Boolean cl){
		
		String[] s1 = (recordList1.get(p.x)).split(";",0);
		String[] s2 = recordList2.get(p.y).split(";",0);
		ArrayList<Double> featList = new ArrayList<Double>();
		
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		String rec1 = "1 ",rec2="2 ";	
		//System.out.println(p.x+" "+p.y);
		for (int i=1;i<8;i++) {
			
			if(s1[i].equals("") || s2[i].equals("")) {
				featList.add(0.0);

//				continue;
			}else{
				rec1+= s1[i]+" ";
				rec2+= s2[i]+" ";
				//System.out.println(i+"and"+s1[i]+"and"+s2[i]);
				Double val = Weightedjaccardold(GetDict("1 "+s1[i]),GetDict("2 "+s2[i]));
				if(val.isNaN())
					val=0.0;
				featList.add(val);
			}
//			System.out.println(recordList1.get(p.x)+" "+s1);
			HashMap<String,Double> CurrFeatMap;
			if (i==1) {
				CurrFeatMap  = Get_small_string_features(s1[i],s2[i]);
				//System.out.println(i+" "+TitleFeatMap);
			}else if(i==2 || i==3) {
				CurrFeatMap  = Get_Single_Word_features(s1[i],s2[i]);
			}else {
				CurrFeatMap  = Get_Numeric_features(s1[i],s2[i]);
			}
			
			for (String s:CurrFeatMap.keySet()) {
				featMap.put(Integer.toString(i)+s,CurrFeatMap.get(s) );
			}
			
		}
		
	 		//featMap.put("OverallJacc",Weightedjaccardold(GetDict(rec1),GetDict(rec2)));
	 		featMap.put("title", (featList.get(0)));
	 		featMap.put("authors", (featList.get(1)));
	 		featMap.put("year", (featList.get(2)));
	 		featMap.put("journal", (featList.get(3)));
	 		//featMap.put("volume", (featList.get(4)));
	 		//featMap.put("pages", (featList.get(5)));
	 		//featMap.put("misc", (featList.get(6)));
	 		if (cl)
	 			featMap.put("Class", 1.0);
	 		else
	 			featMap.put("Class", 0.0);
	 		//if(featMap.get("title")==1  && featMap.get("1title").isNaN())	
			//	System.out.println("something is wrong?"+recordList1.get(p.x)+" "+recordList1.get(p.x));
	 		featMap.put("id1",(double)p.x);	
	 		featMap.put("id2",(double)p.y);	
		return featMap;
	}
	public void topkpernode(int k,ArrayList<double[]> SimList){
		ArrayList<Integer> list = new ArrayList<Integer>();
	        for (int i=1; i<recordList1.size(); i++) {
        		    list.add(new Integer(i));
        	}
	        Collections.shuffle(list);
		ArrayList<Integer> randomList = new ArrayList<Integer>();
		HashMap<Integer,Boolean> found_pair = new HashMap<Integer,Boolean>();
      		  for (int i=0; i<k; i++) {
	            randomList.add(list.get(i));
			found_pair.put(list.get(i),false);
        	}
		int gr=0;
		for(int i=0;i<SimList.size();i++){
			double[] curr = SimList.get(i);
			if(found_pair.containsKey((int)curr[1]) ){
				if(!found_pair.get((int)curr[1])){
					found_pair.put((int)curr[1],true);
					pair p=new pair((int)curr[1],(int)curr[2]);
					if(goldMap.containsKey(p)){
						gr++;
						actualqueried_edge_map.put(p,true);
					}else
						actualqueried_edge_map.put(p,false);
				}

			}

		}
		System.out.println(gr+" number of green edges here");


	}	
	public ArrayList<HashMap<String,Double[]>> TrainClassifier(HashMap<pair,Boolean> training) throws Exception {
		
		int trainingSetsize=training.keySet().size();
		 
		pair tmp=new pair(1,1);
		for (pair p1:training.keySet()) {
			tmp=p1;
			break;
		}
		
		
		HashMap<String,Double> samplefeat = Extractfeat(tmp,training.get(tmp));
		
		featNumbers = samplefeat.keySet().size()-2;
		fvWekaAttributes = new FastVector(featNumbers);
		
		ArrayList<Attribute> feature_lst = new ArrayList<Attribute>();
		HashMap<String,Integer> feature_loc = new HashMap<String,Integer>();
		featnames.clear();
		for(String s: samplefeat.keySet()) {
			if (s.equals("Class") || s.equals("id1") || s.equals("id2"))
				continue;
			
			Attribute feature = new Attribute(s);
			fvWekaAttributes.add(feature);
			feature_loc.put(s, fvWekaAttributes.size()-1);
			featnames.add(s);
		}
		
		
		 /*Attribute feature1 = new Attribute("title");
		 Attribute feature2 = new Attribute("authors");
		 Attribute feature3 = new Attribute("year");
		 Attribute feature4 = new Attribute("journal");
		 Attribute feature5 = new Attribute("volume");
		 Attribute feature6 = new Attribute("pages");
		 Attribute feature7 = new Attribute("misc");
		 Attribute feature8 = new Attribute("sim");
		 */
		 FastVector classLabel = new FastVector(2);
		 classLabel.addElement("red");
		 classLabel.addElement("green");
		 Attribute ClassAttribute = new Attribute("Class", classLabel);
		 
		 // Declare the feature vector
		  
		/* fvWekaAttributes.addElement(feature1);
		 fvWekaAttributes.addElement(feature2);
		 fvWekaAttributes.addElement(feature3);
		 fvWekaAttributes.addElement(feature4);
		 fvWekaAttributes.addElement(feature5);
		 fvWekaAttributes.addElement(feature6);
		 fvWekaAttributes.addElement(feature7);
		 fvWekaAttributes.addElement(feature8);
		
		*/
		 fvWekaAttributes.addElement(ClassAttribute);
		 
			// Create an empty training set
		 TrainingSet = new Instances("Rel", fvWekaAttributes, trainingSetsize);
		 // Set class index
		 TrainingSet.setClassIndex(featNumbers-1);
			 
			 // Create the instance
			for(pair p:training.keySet()) {
				 Instance curr = new DenseInstance(featNumbers);
				//System.out.println(p.x+" "+p.y);
                        	//System.out.println(recordList1.get(p.x)+" change "+recordList1.get(p.y));
                        
				 HashMap<String,Double> feat = Extractfeat(p,training.get(p));
				 //System.out.println(feat);
				 //ArrayList<String> lst = Intersect(recordList.get(p.x), recordList.get(p.y));
				 //System.out.println(p.x+" "+p.y+" "+features.get("jaccard")+" "+features.get("weightedJaccard")+" "+features.get("Cosine")+" "+trainingSet.get(p));
				 for (String s: feat.keySet()) {
					 if (s.equals("Class") || s.equals("id1") || s.equals("id2"))
							continue;
					 if(!feat.get(s).isNaN())
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(s)), feat.get(s));
					 else
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(s)), 0.0);
				 }
				/* if (feat.get("title") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(0), feat.get("title"));
				 if (feat.get("authors") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(1), feat.get("authors"));
				 if (feat.get("year") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(2), feat.get("year"));
				 if (feat.get("journal") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(3), feat.get("journal"));
				 if (feat.get("volume") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(4), feat.get("volume"));
				 if (feat.get("pages") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(5), feat.get("pages"));
				 if (feat.get("misc") >= 0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(6), feat.get("misc"));
				 if(edge_prob.containsKey(p))
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(7), edge_prob.get(p));
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(7), 0.0);
				 */
				 if(feat.get("Class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 TrainingSet.add(curr);
			 }
			 
			
			  
			 Classifier cModel  =new RandomTree();// new J48();//new Logistic();//new J48();
			// cModel.SetSeed((int)generator.nextDouble()*Integer.MAX_VALUE);
			// String[] opt = {"depth 5"};
			 //cModel.SetDepth(10);
			
			 
			 cModel.buildClassifier(TrainingSet);
			 System.out.println(cModel.toString()+"model is ");
			
			 
			 Tree t11 = cModel.GetRandomTree();
			 //System.out.println(t11.toString(0));
			 ArrayList<HashMap<String,Double[]>> rulelst = getRules(t11);
			
			 
			 
			 System.out.println(rulelst.size()+"Number of rules ");
			 for(HashMap<String,Double[]> r : rulelst) {
				 System.out.println("New Rule ");
				 for(String s:r.keySet()) {
					 System.out.print(s+" "+r.get(s)[0]+" "+r.get(s)[1]+" ;;");
				 }
			 }
			 
			 
			 
			
			 return rulelst;
	}
	
	public ArrayList<ArrayList<HashMap<String,Double[]>>> GetRules(HashMap<pair,Boolean> training) throws Exception {
		for(pair p:actualqueried_edge_map.keySet()) {
			training.put(p, actualqueried_edge_map.get(p));
		}


		//Get green and red and add more green here
		//
		int gr=0,rd=0;
		for(pair p:training.keySet()){
			if(training.get(p))
				gr++;
			else
				rd++;
		}
/*
		for(int i=0;i<rd-gr;i++){
			pair p = new pair(i,i);
			training.put(p,true);
			actualqueried_edge_map.put(p,true);
		}

*/		int numClassifiers=10;
		int datasetSize=(int)(training.size()*0.6);
		System.out.println("Dataset size "+datasetSize+" "+training.keySet().size());
		ArrayList<ArrayList<HashMap<String,Double[]>>> rulelstlst = new ArrayList<ArrayList<HashMap<String,Double[]>>>();
		for(int i=0;i<10;i++) {
			HashMap<pair,Boolean>smallTraining = new HashMap<pair,Boolean>();
			ArrayList<pair> keylst=new ArrayList(training.keySet());
			Collections.shuffle(keylst,new Random(rand.nextInt()));
			for(int j=0;j<datasetSize;j++)
				smallTraining.put(keylst.get(j), training.get(keylst.get(j)));
			ArrayList<HashMap<String,Double[]>> rulelst = TrainClassifier(smallTraining);
			//Check if the classifier is good else ask more questions
			System.out.println(smallTraining.keySet().size()+" "+smallTraining.keySet());
			rulelstlst.add(rulelst);
		}
		return rulelstlst;
	}
	public boolean satisfyRule(HashMap<String,Double[]> rule, HashMap<String,Double> feat) {
		// boolean holds=true;
		for(String s:rule.keySet()) {
			 if(s.trim().equals("class"))
				 continue;
			 String s1 = s.split(";",0)[0];
			 
			/* for(String s2: feat.keySet())
				 System.out.println(s2+" "+feat.get(s2));
			 for(String s2: rule.keySet())
				 System.out.println(s2+" "+rule.get(s2));
			  */
			if(!feat.containsKey(s1.trim()))
				return false;
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
	public double Test( pair p,Boolean cl, ArrayList<ArrayList<HashMap<String,Double[]>>>  Rulelst) throws Exception {
			//System.out.println(p.x+" "+p.y);
			//System.out.println(recordList1.get(p.x)+" change "+recordList1.get(p.y));
			HashMap<String,Double> feat = Extractfeat(p,cl);
			if (edge_prob.containsKey(p))
				feat.put("sim", edge_prob.get(p));
			else
				feat.put("sim", 0.0);
			// System.out.println(ModelRules.size()+"Number of rules ");
			 int iter=0,out=0;
			 for(ArrayList<HashMap<String,Double[]>> Tree: Rulelst) {
				 for(HashMap<String,Double[]> r : Tree) {
					 boolean holds=true;
					 if(satisfyRule(r,feat)){
						 /*if(goldMap.get(p.x).equals(goldMap.get(p.y)) && r.get("class")[1]>0)
							 goodcount.set(iter, goodcount.get(iter)+1);
						 else if (!goldMap.get(p.x).equals(goldMap.get(p.y)) && r.get("class")[1]<=0)
							 goodcount.set(iter, goodcount.get(iter)+1);
						 else if  (goldMap.get(p.x).equals(goldMap.get(p.y)) && r.get("class")[1]<=0)
							 badcount.set(iter, badcount.get(iter)+1);
						 else
							 badcount.set(iter, badcount.get(iter)+1);
						*/ 
						 if(r.get("class")[1]>0)
							 out+=r.get("class")[1];
						 break;
					 }
					 iter++;
				 }
			 }
			 
			 return out*1.0/Rulelst.size();
			//return eTest.evaluateModelOnce(cModel, curr);	 
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
	public boolean  TrainClassifierPython(ArrayList<double[]> SimList, int start,int num, boolean newsamples) throws Exception{
		
		String falcon_data = folder+"/falcondata.txt";
		Scanner scanner = new Scanner(new File(falcon_data));

		PrintStream falcon_output = new PrintStream(folder+"/falcon_train.txt");
		String line;
		int it=0;
                 while(scanner.hasNextLine()){
                       line= scanner.nextLine();
			String[] idlist = line.split(",",0);
			HashMap<String,Double> feat = new HashMap<String,Double>();
			pair p = new pair(Integer.parseInt(idlist[0])-1,Integer.parseInt(idlist[1])-1);
			if(idlist[2].equals("t"))
				feat=Extractfeat(p,true);
			else
				feat=Extractfeat(p,false);

			if(it==0){
				for (String s: feat.keySet()){
                                                if(!s.equals("Class"))
                                                        falcon_output.print(s+",");
                                        }
                                        falcon_output.print("Class\n");
			}


			for (String s: feat.keySet()){
                                    if( s.equals("Class"))
                                                        continue;
                                                if(feat.get(s).isNaN())
                                                        falcon_output.print("0.0,");
                                                else falcon_output.print(feat.get(s)+",");

                           }
                                                falcon_output.print(feat.get("Class")+"\n");
			it+=1;
		}
		falcon_output.close();

        HashMap<Integer,Integer> blocking_edges = new HashMap<Integer,Integer>();
		
		PrintStream classifier_output = new PrintStream(folder+"/classifier.txt");
		PrintStream known_output = new PrintStream(folder+"/known.txt");
		PrintStream unseen_output = new PrintStream(folder+"/unseen.txt");
		for(int i=start;;i++) {
		//for(int i=start;i<start+num;i++) {

                        {
                                if(i>=SimList.size())
                                        break;
				HashMap<String,Double> feat = new HashMap<String,Double>();
                                double[] curr = SimList.get(i);
                                pair p = new pair((int)curr[1],(int)curr[2]);
                                if(!edge_prob.containsKey(p))
                                        continue;
                                if(goldMap.containsKey(p))
                                        feat=(Extractfeat(p, true));
                                else
                                        feat=(Extractfeat(p, false));
				if(feat.keySet().size()<15)
					System.out.println("something is weong"+feat+" "+feat.keySet());
				
				if (i==start){
					for (String s: feat.keySet()){
						if(!s.equals("Class"))
							classifier_output.print(s+",");
						if(!s.equals("Class"))
							known_output.print(s+",");
						if(!s.equals("Class"))
							unseen_output.print(s+",");
					}
					classifier_output.print("Class\n");
					known_output.print("Class\n");
					unseen_output.print("Class\n");
				}
				 if(blocking_edges.containsKey(p.x))
                                        blocking_edges.put(p.x,blocking_edges.get(p.x)+1);
                                else
                                        blocking_edges.put(p.x,1);
                                if(blocking_edges.containsKey(p.y))
                                        blocking_edges.put(p.y,blocking_edges.get(p.y)+1);
                                else
                                        blocking_edges.put(p.y,1);
                                if(blocking_edges.get(p.x)>100 || blocking_edges.get(p.y)>100){
                                        ;//i++;
                                        continue;
                                        //
                                   }
                                        //
                                        //
                                        //

				if (i<=start+num){
					for (String s: feat.keySet()){
						if( s.equals("Class"))
							continue;
						if(feat.get(s).isNaN())
							classifier_output.print("0.0,");
						else classifier_output.print(feat.get(s)+",");
	
					}
						classifier_output.print(feat.get("Class")+"\n");
				}else{
					for (String s: feat.keySet()){
						if( s.equals("Class"))
							continue;
						if(feat.get(s).isNaN())
							unseen_output.print("0.0,");
						else unseen_output.print(feat.get(s)+",");
	
					}
						unseen_output.print(feat.get("Class")+"\n");


				}
			}
		}
		classifier_output.close();
		unseen_output.close();
		System.out.println(actualqueried_edge_map.keySet().size());
		for(pair p:actualqueried_edge_map.keySet()){
			System.out.println(actualqueried_edge_map.get(p)+" "+p.x+" "+p.y);
			if(p.x>=p.y)
				continue;
			HashMap<String,Double> feat = new HashMap<String,Double>();
			if(goldMap.containsKey(p))
                                        feat=(Extractfeat(p, true));
                                else
                                        feat=(Extractfeat(p, false));
			for (String s: feat.keySet()){
				if( s.equals("Class"))
					continue;
				if(feat.get(s).isNaN())
					known_output.print("0.0,");
				else known_output.print(feat.get(s)+",");
			}
				
			known_output.print(feat.get("Class")+"\n");
			
		}
		known_output.close();
		 //Runtime.getRuntime().exec(new String[] {"cd ~/javaCode/src/"+folder+" ; python learner.py; cd ../"});
		//Call python code here
		//
		String[] args1 = new String[] {"/bin/bash", "-c", "cd ~/javaCode/src/songsClassifier; python learner.py > o1; cd .."};
		Process proc = new ProcessBuilder(args1).start();
		 proc.waitFor();
		String pythonoutput = folder+"/inferred.txt";
		String pythonqueried = folder+"/queried.txt";
		String newoutput = folder+"/newoutput.txt";
		scanner = new Scanner(new File(pythonoutput));

                 while(scanner.hasNextLine()){
                       line= scanner.nextLine();
			String[] idlist = line.split(" ",0);
			pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
			pair p1 = new pair(Integer.parseInt(idlist[1]),Integer.parseInt(idlist[0]));
			inferred_edges.put(p,true);
			inferred_edges.put(p1,true);
			System.out.println("inferred"+p.x+" "+p.y);
		}
		
		scanner = new Scanner(new File(newoutput));
                 while(scanner.hasNextLine()){
                       line= scanner.nextLine();
			String[] idlist = line.split(" ",0);
			pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
			pair p1 = new pair(Integer.parseInt(idlist[1]),Integer.parseInt(idlist[0]));
			inferred_edges.put(p,true);
			inferred_edges.put(p1,true);
			//System.out.println("inferred"+p.x+" "+p.y);
		}
		
		scanner = new Scanner(new File(pythonqueried));
                 while(scanner.hasNextLine()){
                       line= scanner.nextLine();
			String[] idlist = line.split(" ",0);
			pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
			pair p1 = new pair(Integer.parseInt(idlist[1]),Integer.parseInt(idlist[0]));
			if(idlist[2].equals("true")){
				actualqueried_edge_map.put(p,true);
				actualqueried_edge_map.put(p1,true);
			}else{
				actualqueried_edge_map.put(p,false);
				actualqueried_edge_map.put(p1,false);
			}
			System.out.println("queried"+p.x+" "+p.y);
		}
	

		return true;
	}
	public ArrayList<ArrayList<HashMap<String,Double[]>>> TrainFalconClassifier(ArrayList<double[]> SimList, int start,int num, boolean newsamples) throws Exception{
		
		ArrayList<HashMap<String,Double>> featureLst = new ArrayList<HashMap<String,Double>>();
		ArrayList<Integer> AlreadyQueried = new ArrayList<Integer>();
		
		for(int i=start;i<start+num;i++) {
			
			{
				if(i>=SimList.size())
					break;
				
				double[] curr = SimList.get(i);
				pair p = new pair((int)curr[1],(int)curr[2]);
				if(!edge_prob.containsKey(p))
					continue;
				if(goldMap.containsKey(p))
					featureLst.add(Extractfeat(p, true));
				else
					featureLst.add(Extractfeat(p, false));
			//	if(actualqueried_edge_map.containsKey(p))
			//		AlreadyQueried.add(featureLst.size()-1);
			}
		}
		Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		Collections.shuffle(featureLst,newgen);

		int Vsize = (int) (0.03*featureLst.size());
		ArrayList<HashMap<String,Double>> V = new ArrayList<HashMap<String, Double>>( featureLst.subList(0, Vsize));
		ArrayList<HashMap<String,Double>> NewfeatureLst = new ArrayList<HashMap<String, Double>>( featureLst.subList(Vsize, featureLst.size()));
		for(pair p:actualqueried_edge_map.keySet()) {
			if(p.x>p.y) {
				if(goldMap.containsKey(p))
					NewfeatureLst.add(Extractfeat(p, true));
				else
					NewfeatureLst.add(Extractfeat(p, false));
				AlreadyQueried.add(NewfeatureLst.size()-1);
			}
		}
		if(AlreadyQueried.size()==0) {
			for(int i=0;i<20;i++) {
				AlreadyQueried.add(i);
				pair tmp = new pair(NewfeatureLst.get(i).get("id1").intValue(),NewfeatureLst.get(i).get("id2").intValue());
				pair tmp1 = new pair(NewfeatureLst.get(i).get("id2").intValue(),NewfeatureLst.get(i).get("id1").intValue());
				if(goldMap.containsKey(tmp)||goldMap.containsKey(tmp1)) {
					actualqueried_edge_map.put(tmp, true);
					actualqueried_edge_map.put(tmp1, true);

				}
				else {
					actualqueried_edge_map.put(tmp, false);
					actualqueried_edge_map.put(tmp1, false);
				}
			}
		}
		
		boolean Blocking=false;
		return al_matcher(NewfeatureLst,  AlreadyQueried,  V,  Blocking);
		
	}		
	public ArrayList<ArrayList<HashMap<String,Double[]>>> al_matcher(ArrayList<HashMap<String,Double>> featureLst, ArrayList<Integer> AlreadyQueried, ArrayList<HashMap<String,Double>> V, boolean Blocking) throws Exception {
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
			 FastVector fvWekaAttributes = new FastVector(featNumbers);
			 featnames.clear();
			 HashMap<String,Integer> feature_loc = new HashMap<String,Integer>();
			 for(String s:featureLst.get(0).keySet()) {
				 if(s.equals("Class")|| s.equals("id1") || s.equals("id2"))
					 continue;
				 
				 Attribute tmp = new Attribute(s);
				 features.add(tmp);
				 fvWekaAttributes.addElement(tmp);
				 feature_loc.put(s, fvWekaAttributes.size()-1);
					featnames.add(s);
					
			 }
			 FastVector classLabel = new FastVector(2);
			 classLabel.addElement("red");
			 classLabel.addElement("green");
			 Attribute ClassAttribute = new Attribute("class", classLabel);
			 

			
			
			 fvWekaAttributes.addElement(ClassAttribute);
			 
				
			ArrayList<Instances> TrainingSetList = new ArrayList<Instances>();
			ArrayList<Classifier> ModelLst= new ArrayList<Classifier>();
			 ArrayList<ArrayList<HashMap<String,Double[]>>> ruleLstlst= new  ArrayList<ArrayList<HashMap<String,Double[]>>>();
			for(int iter=0;iter<numClassifiers;iter++) {
				
			
			 Instances TrainingSet;
	
			 
			 TrainingSet = new Instances("Rel", fvWekaAttributes, datasetSize);
			 TrainingSet.setClassIndex(featNumbers-1);
			 
			 Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
			Collections.shuffle(AlreadyQueried,newgen);
				
			 for(int k=0;k<Math.min(AlreadyQueried.size(), datasetSize);k++) {
				 int recid = AlreadyQueried.get(k);
				 DenseInstance curr = new DenseInstance(featNumbers);
				 HashMap<String,Double>rowfeat= featureLst.get(recid);
				  for(String feat:rowfeat.keySet()) {
					 
					  if(feat.equals("Class")|| feat.equals("id1") || feat.equals("id2"))
						  continue;
					  if(!rowfeat.get(feat).isNaN())
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), rowfeat.get(feat));
					  else
						  curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), 0.0);
					
				 }
				 if(rowfeat.get("Class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 TrainingSet.add(curr);
			 }
			 
			 
			  
			 Classifier cModel =new RandomTree();
			 cModel.SetSeed((int)generator.nextDouble()*Integer.MAX_VALUE);
			 
			 cModel.buildClassifier(TrainingSet);
			 Tree t11 = cModel.GetRandomTree();
			
			 ArrayList<HashMap<String,Double[]>> rulelst = getRules(t11);
			 	ruleLstlst.add(rulelst);
			 	TrainingSetList.add(TrainingSet);
			 	ModelLst.add(cModel);
			}
			ListofTrainingSetList.add(TrainingSetList);
			ListofModelLst.add(ModelLst);
			ListofruleLstlst.add(ruleLstlst);
			 
			
			 
			 boolean classificationNeeded=true;
			 
			 double totalConfidence=0.0;
			 for(int i=0;i<V.size();i++) {
				 HashMap<String,Double>rowfeat= V.get(i);
				 DenseInstance curr = new DenseInstance(featNumbers);
				 
				 for(String feat:rowfeat.keySet()) {
					 
					  if(feat.equals("Class")|| feat.equals("id1") || feat.equals("id2"))
						  continue;
					  if(!rowfeat.get(feat).isNaN())
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), rowfeat.get(feat));
					  else
						  curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), 0.0);
					
				 }
				 if(rowfeat.get("Class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 
				 int g=0;
				 for(int j=0;j<numClassifiers;j++) {
					 curr.setDataset(TrainingSetList.get(j));
					 double[] fDistribution = ModelLst.get(j).distributionForInstance(curr);
					 if(fDistribution[0]<0.5)
						 g++;
					 
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
			 
			 if (processConfidenceList(confidenceLst,window,nconverged,nhigh,ndegrade,epsilon))
				 break;
			 System.out.println(totalConfidence+" confidence"+V.size()+" "+confidenceLst.size()+" "+AlreadyQueried.size());
			 for(int i=0;i<confidenceLst.size();i++)
				 System.out.println("confidence" +i+" "+confidenceLst.get(i));

			// if(!Blocking && totalConfidence>0.9)
		//		 break;
			 if (Blocking &&totalConfidence==1.0)
				 break;
			 
			 
			 
			 
			 if(classificationNeeded) {
				 
				 ArrayList<double[]> confusingLst = new ArrayList<double[]>();
				 for(int i=0;i<featureLst.size();i++) {
					 HashMap<String,Double>rowfeat= featureLst.get(i);
					 DenseInstance curr = new DenseInstance(featNumbers);
					 
					 for(String feat:rowfeat.keySet()) {
						  if(feat.equals("Class")|| feat.equals("id1") || feat.equals("id2"))
							  continue;
						  if(!rowfeat.get(feat).isNaN())
							 curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), rowfeat.get(feat));
						  else
							  curr.setValue((Attribute)fvWekaAttributes.elementAt(feature_loc.get(feat)), 0.0);
						
					 }
					 if(rowfeat.get("Class")>0)
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
					 else
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
					 
					 int g=0;
					 for(int j=0;j<numClassifiers;j++) {
						 curr.setDataset(TrainingSetList.get(j));
						 double[] fDistribution = ModelLst.get(j).distributionForInstance(curr);
						 if(fDistribution[0]<0.5)
							 g++;
						 
						 
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
									if(goldMap.containsKey(tmp) || goldMap.containsKey(tmp1)){
									 actualqueried_edge_map.put(tmp, true);//goldMap.get(tmp.x).equals(goldMap.get(tmp.y)));
									 actualqueried_edge_map.put(tmp1, true);//goldMap.get(tmp1.x).equals(goldMap.get(tmp1.y)));
									}
									else{
									 actualqueried_edge_map.put(tmp, false);//goldMap.get(tmp.x).equals(goldMap.get(tmp.y)));
									 actualqueried_edge_map.put(tmp1, false);//goldMap.get(tmp1.x).equals(goldMap.get(tmp1.y)));

									}
									 System.out.println("queries is "+queries);
								 }
								 
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
			 
			
			 

		}
		
		
		 System.out.println(ListofruleLstlst.get(ListofruleLstlst.size()-1).size()+"Number of rules ");
		
		 
		 
		return ListofruleLstlst.get(ListofruleLstlst.size()-1);
		
		 
	}

	
	public ArrayList<ArrayList<HashMap<String,Double[]>>> TrainClassifier(ArrayList<double[]> SimList, int start,int num, boolean newsamples) throws Exception {
		ArrayList<ArrayList<pair>> simpartitions = new ArrayList<ArrayList<pair>>();
		for(int i=0;i<10;i++) {
			ArrayList<pair> part = new ArrayList<pair>();
			simpartitions.add(part);
		}
	
		
		double max = SimList.get(start)[0]+0.00000001;
			double min = -1;
			if(start+num < SimList.size())min =  SimList.get(start+num)[0];
			else min=0;
		System.out.println(simpartitions.size()+" "+max+" "+min);
		//if(newsamples) 
			
			
			ArrayList<double[]> problist = new ArrayList<double[]>();
			int j=0;
			for(int i=start;i<start+num;i++) {
				//for(int j=i+1;j<highexpectedsize.size();j++)
				{
					if(i>=SimList.size())
						break;
					double[] curr = SimList.get(i);
					pair p = new pair((int)curr[1],(int)curr[2]);
					if(!edge_prob.containsKey(p))
						continue;
					//int prob=(int)(10*(edge_prob.get(p)-0.00000001));
					int prob=(int)(10*(edge_prob.get(p)-min)*1.0/(max-min));
					//prob = (int)((i-start)*10.0/num);
					ArrayList<pair> tmp = simpartitions.get(prob);
					tmp.add(p);
					//System.out.println("loc"+prob);
					simpartitions.set(prob, tmp);
				}
			}
			System.out.println(simpartitions.size());
			
		
		
		
		//Form a training set here
		
		training = new HashMap<pair,Boolean>();
		ArrayList<Integer>validationCounter = new ArrayList<Integer>();
		System.out.println(simpartitions.size());
		int found_green=0;
		int numperpartition=10;
		if(!newsamples)
			numperpartition=0;
		for(int i=0;i<simpartitions.size();i++) {
			ArrayList<pair> tmp = simpartitions.get(i);
			System.out.println(tmp.size()+" partition number"+i);
			Collections.shuffle(tmp);
			for(int i1=0;i1<numperpartition;i1++) {
				if(i1>=tmp.size())
					break;
				pair p=tmp.get(i1);
				pair p1=new pair(p.y,p.x);
				if(goldMap.containsKey(p)){
					training.put(p, true);
					actualqueried_edge_map.put(p,true);
					actualqueried_edge_map.put(p1,true);
					found_green++;
				}
				else{
					actualqueried_edge_map.put(p,false);
					actualqueried_edge_map.put(p1,false);
					training.put(p, false);
				}
			}
			validationCounter.add(numperpartition);
		}
/*
		for(pair p:actualqueried_edge_map.keySet()){

			training,.put(p,actualqueried_edge_map.get(p));
		}
*/		
		if(found_green<20 && newsamples){
			for(int i=0;i<simpartitions.get(0).size();i++){
				double[] curr = SimList.get(i);
				pair p = new pair((int)curr[1],(int)curr[2]);
				//pair p = simpartitions.get(0).get(i);//new pair((int)curr[1],(int)curr[2]);
				//pair p1 = new pair(p.y,p.x);//((int)curr[2],(int)curr[1]);
				pair p1 = new pair((int)curr[2],(int)curr[1]);
				if(actualqueried_edge_map.containsKey(p))
					continue;
				if(goldMap.containsKey(p)){
                                        training.put(p, true);
                                        actualqueried_edge_map.put(p,true);
                                        actualqueried_edge_map.put(p1,true);
                                        found_green++;
                                }
                                else{
                                        actualqueried_edge_map.put(p,false);
                                        actualqueried_edge_map.put(p1,false);
                                        training.put(p, false);
                                }
				if(found_green>20)
				break;
			}
		}
		
		ArrayList<ArrayList<HashMap<String,Double[]>>> rulelstlst = new ArrayList<ArrayList<HashMap<String,Double[]>>>();
		while(true) {
			 rulelstlst = GetRules(training);
			double total=0,total_confidence=0.0;
			int numq=0;
			int gtscore=0;
			boolean exit=false;
			System.out.println("Testing now");
			for(int i=0;i<simpartitions.size();i++) {
				if(validationCounter.get(i)>=simpartitions.get(i).size())
					continue;
				pair p = simpartitions.get(i).get(validationCounter.get(i));
				Boolean gtoutput = goldMap.containsKey(p);
				double out = Test(p,gtoutput,rulelstlst);
				//total+=out;
				if(gtoutput){
					total_confidence +=out;	
				}else
					total_confidence += 1-out;
				if (out>0.5)
					total+=1;
				else
					total+=0;
				numq+=1;
				if(out<=0.5 && gtoutput){
					for(int j1=validationCounter.get(i);j1<validationCounter.get(i)+5;j1++){
						pair t = simpartitions.get(i).get(j1);
						 Boolean gtoutput1 = goldMap.containsKey(t);
						actualqueried_edge_map.put(t, gtoutput1);
					}
					validationCounter.set(i, validationCounter.get(i)+5);
				//	actualqueried_edge_map.put(p, gtoutput);
				}else if(out>=0.5 && !gtoutput){
					for(int j1=validationCounter.get(i);j1<validationCounter.get(i)+5;j1++){
						pair t = simpartitions.get(i).get(j1);
						 Boolean gtoutput1 = goldMap.containsKey(t);
						actualqueried_edge_map.put(t, gtoutput1);
					}
					validationCounter.set(i, validationCounter.get(i)+5);
					//validationCounter.set(i, validationCounter.get(i)+1);
					//actualqueried_edge_map.put(p, gtoutput);

				}
				if(gtoutput)
					gtscore++;
			}
			System.out.println("total score is "+total*1.0/(numq)+" "+numq+" "+gtscore+" "+simpartitions.size()+" "+total_confidence*1.0/numq);
			if((Math.abs((gtscore-total)*1.0/numq) <= 0.1)) {
				exit=true;
			}/*
			int low=0;
			ArrayList<pair> lowpairs =new ArrayList<pair>();
			for(int i=0;i<simpartitions.size();i++){
				for(int j1=0;j1<simpartitions.get(i).size();j1++){
					pair p = simpartitions.get(i).get(j1);
					 Boolean gtoutput = goldMap.containsKey(p);
		                         double out = Test(p,gtoutput,rulelstlst);
					if(out<0.6 && out>0.4){
						low++;//System.out.println(out+" "+p.x+" "+p.y+" "+gtoutput);
						lowpairs.add(p);
					}
				}
			}*//*
			if(low>100){
				for(int i=0;i<20;i++){
					if (goldMap.containsKey(lowpairs.get(i)))
						training.put(lowpairs.get(i),true);
					else
						training.put(lowpairs.get(i),false);

				}
				exit=false;
			}*/
				
			//System.out.println(low+" low confidence pairs");

		if(exit || training.keySet().size()>220)
			break;
			
		}
	/*
		ArrayList<ArrayList<HashMap<String,Double[]>>> rulelstlst = GetRules(training);
		double total=0;
		int numq=0;
		System.out.println("Testing now");
		for(int i=0;i<simpartitions.size();i++) {
			if(validationCounter.get(i)>=simpartitions.get(i).size())
				continue;
			pair p = simpartitions.get(i).get(validationCounter.get(i));
			Boolean gtoutput = goldMap.containsKey(p);
			double out = Test(p,gtoutput,rulelstlst);
			total+=out;
			numq+=1;
		}
		System.out.println("total score is "+total*1.0/(numq)+" "+numq+" "+simpartitions.size());
	*/	//pair p,Boolean cl, ArrayList<ArrayList<HashMap<String,Double[]>>>  Rulelst
		return rulelstlst;
	}
	
	public void processBlocks() throws Exception{
		//String[] args1 = new String[] {"/bin/bash", "-c", "python /home/sainyam/javaCode/src/songsClassifier/learner.py > o1"};
		//Process proc = new ProcessBuilder(args1).start();
		 //proc.waitFor();
				final long startTime = System.currentTimeMillis();
		if (mergeBlocks()==0) {
			System.out.println("done merging");
			//return;
		}
			
		//String ttmp = "cell!block!nine!new";
	//	System.out.println("testing"+BlockTreeMap1.get(3).get(ttmp));
		
		recallprint= new PrintStream(folder+"/recall.txt");
	//	PrintStream Nodeorder= new PrintStream("nodeorder.txt");
		
//		HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

		int blockId=0;
		HashMap<String,Integer> blockSize = new HashMap<String,Integer>();

		float totalpairs=(float) (Math.log(recordList1.keySet().size())+ Math.log(recordList2.keySet().size()));//recordList1.keySet().size()*recordList2.keySet().size();
		blockWeight1.clear();
		for(int i=0;i<BlockTreeMap1.keySet().size();i++) {
			HashMap<String,ArrayList<Integer>> level = BlockTreeMap1.get(i);
			for(String s:level.keySet()) {
				//if (i==0) 
				{
					
					//if(!BlockTreeMap2.get(i).containsKey(s))
					//	continue;
					//if (BlockTreeMap2.get(i).get(s).size()==0){
					//	System.out.println("problem "+s);
					//	return;
					//}
					if(s.equals("a"))
					System.out.println("it is present!");
					blockSize.put(s,level.get(s).size());
					double score1 = recordList1.keySet().size()*1.0/level.get(s).size();
					double score2 = score1;//recordList2.keySet().size()*1.0/BlockTreeMap2.get(i).get(s).size();
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
		
	/*	HashMap<Integer,Integer> first_edge = new HashMap<Integer,Integer>();
		HashMap<Integer,Double> first_round = new HashMap<Integer,Double>();
		for(int i=0;i<SimilarityList.size();i++){
			double[] curr = SimilarityList.get(i) ;
                        boolean out =false;
                        pair p1 = new pair((int)curr[1],(int)curr[2]);
                        pair p2 = new pair((int)curr[2],(int)curr[1]);
			out = goldMap.containsKey(p1);

			if(first_round.containsKey(p1.x) && first_round.containsKey(p1.y))
				continue;
			else if (first_round.containsKey(p1.x)){
				if(out)
					first_round.put(p1.y,curr[0]);
				if (first_edge.containsKey(p1.y))
					first_edge.put(p1.y,first_edge.get(p1.y)+1);
				else
					first_edge.put(p1.y,1);
				continue;
			}else if (first_round.containsKey(p1.y)){

				if(out)
					first_round.put(p1.x,curr[0]);
				if (first_edge.containsKey(p1.x))
					first_edge.put(p1.x,first_edge.get(p1.x)+1);
				else
					first_edge.put(p1.x,1);
				continue;
			}else{

				if(out){
					first_round.put(p1.y,curr[0]);
					first_round.put(p1.x,curr[0]);

				}
			}
			if (first_edge.containsKey(p1.x))
				first_edge.put(p1.x,first_edge.get(p1.x)+1);
			else
				first_edge.put(p1.x,1);
			if (first_edge.containsKey(p1.y))
				first_edge.put(p1.y,first_edge.get(p1.y)+1);
			else
				first_edge.put(p1.y,1);
			

		}

		PrintStream first = new PrintStream("first.txt");
		for(Integer u:first_round.keySet()){
			first.println(u+" "+first_edge.get(u)+" "+first_round.get(u));
		}
		if(first_edge.keySet().size()>0)
			return;
	*/	/*for(int i=0;i<1000;i++) {
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

		
	Random randomno = new Random();	
		System.out.println(edge_prob.keySet().size()+"edgeprob"+" "+SimilarityList.size()+" "+(edge_prob.keySet().size()));
		
		int feedback_val=100000;
		while(actualqueried_edge_map.keySet().size()<=4) {
			pair  randomKey = (pair) edge_prob.keySet().toArray()[new Random().nextInt(edge_prob.keySet().size())];
			pair  randomKey1 = new pair(randomKey.y,randomKey.x);
			if(!goldMap.containsKey(randomKey)) {
				actualqueried_edge_map.put(randomKey, false);
				actualqueried_edge_map.put(randomKey1, false);
			}
			if(actualqueried_edge_map.keySet().size()==4)
				break;
		}
		while(actualqueried_edge_map.keySet().size()<=8) {
			pair  randomKey = (pair) goldMap.keySet().toArray()[new Random().nextInt(goldMap.keySet().toArray().length)];
			pair  randomKey1 = new pair(randomKey.y,randomKey.x);
			if(goldMap.containsKey(randomKey)) {
				actualqueried_edge_map.put(randomKey, true);
				actualqueried_edge_map.put(randomKey1, true);
			}
			if(actualqueried_edge_map.keySet().size()==8)
				break;
		}
//		topkpernode(100, SimilarityList);

	/*	
		PrintStream dataset_print1= new PrintStream(folder+"/pos.txt");
		PrintStream dataset_print2= new PrintStream(folder+"/neg.txt");
		ArrayList<String> featureOrder = new ArrayList<String>();
		for (pair p:edge_prob.keySet()){
			HashMap<String,Double>  dict= Extractfeat(p,goldMap.containsKey(p));
			if(featureOrder.size()==0){
				featureOrder.addAll(dict.keySet());
				dataset_print1.println(featureOrder);
				dataset_print2.println(featureOrder);
			}
			if(dict.get("Class")>0){

			for (String s: dict.keySet()){
				if(s.equals("Class"))
					continue;
				else
				dataset_print1.print(dict.get(s)+",");
			}
			dataset_print1.print(dict.get("Class")+"\n");


			}else{

			for (String s: dict.keySet()){
				if(s.equals("Class"))
					continue;
				else
				dataset_print2.print(dict.get(s)+",");
			}
			dataset_print2.print(dict.get("Class")+"\n");


			}
		}
		
	*//*	HashMap<pair,Boolean> pos = new HashMap<pair,Boolean>();
		HashMap<pair,Boolean> neg = new HashMap<pair,Boolean>();

		for (pair p:edge_prob.keySet()){
			pair p1 = new pair(p.y,p.x);
			if(actualqueried_edge_map.keySet().size()>=400)
				break;
			boolean out = goldMap.containsKey(p);
			if(out && pos.keySet().size()>100)
				continue;
			else if (out){
				actualqueried_edge_map.put(p,true);
				actualqueried_edge_map.put(p1,true);
			}
			if(!out && neg.keySet().size()>100)
				continue;
			else if (!out){
				actualqueried_edge_map.put(p,false);
				actualqueried_edge_map.put(p1,false);
			}
		}

*/
		//Choose two random pairs which are green and red	
		//ModelRuleslst = TrainClassifier(SimilarityList, 0, feedback_val, true);
		boolean ooutput = false;//TrainClassifierPython(SimilarityList, 0, feedback_val, true);
		if (useClassifier)
			ooutput = TrainClassifierPython(SimilarityList, 0, feedback_val, true);
		//ModelRuleslst = TrainFalconClassifier(SimilarityList, 0, feedback_val, true);
		if(ooutput)
			return;
		int curr_iter = 0;
		
		HashMap<Integer,Integer> clust_map = new HashMap<Integer,Integer>();
		//HashMap<Integer,ArrayList<Integer>> inverseclust_map = new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,Integer> blocking_edges = new HashMap<Integer,Integer>();
		int actual_comparisons=0;	
		int gr=0, trgr=0;
		while(true){
			
			
			boolean end=false;
			for(int i=curr_iter;i<SimilarityList.size();) {
				if(i==SimilarityList.size())
					end=true;
				
				double[] curr = SimilarityList.get(i);
				int u = (int)curr[1];
				int v= (int) curr[2];
				if(blocking_edges.containsKey(u))
					blocking_edges.put(u,blocking_edges.get(u)+1);
				else
					blocking_edges.put(u,1);
				if(blocking_edges.containsKey(v))
					blocking_edges.put(v,blocking_edges.get(v)+1);
				else
					blocking_edges.put(v,1);

				if(blocking_edges.get(u)>100 || blocking_edges.get(v)>100){
					;//i++;
					//continue;

				}
				else if(u==v){
					;//	i++;
					//continue;
				}else{
				actual_comparisons++;
				boolean[] output =  query_edge_prob( oracle, goldMap, u, v,0,0);
				if(output[1]) {
					if(clust_map.containsKey(u) && clust_map.containsKey(v)) {
						if(!clust_map.get(u).equals(clust_map.get(v))) {
							ArrayList<Integer> c1 = set_clusters.get(clust_map.get(u)).get_component();
							ArrayList<Integer> c2 = set_clusters.get(clust_map.get(v)).get_component();
							gr+= c1.size()*c2.size();
							if(output[0])
								trgr +=  c1.size()*c2.size();
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
							if(output[0])
								trgr +=  tmp.size()-1 ;
						}else if(clust_map.containsKey(v)){
							ArrayList<Integer> tmp = set_clusters.get(clust_map.get(v)).get_component();
							tmp.add(u);
							component c = new component(tmp);
							set_clusters.set(clust_map.get(v),c);
							clust_map.put(u, clust_map.get(v));
							//inverseclust_map.put(set_clusters.size()-1, tmp);
							gr+= tmp.size()-1 ;
							if(output[0])
								trgr +=  tmp.size()-1 ;
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
							if(output[0])
								trgr ++;
						}
					}
				}
				else{
					
					;/*System.out.println(u+" "+v+" "+curr[0]);
					System.out.println(recordList1.get(u));
					System.out.println(recordList1.get(v));
					 */
				}				
				}
				i++;
				if(i%feedback_val == 0) {
					curr_iter=i;
					//Send feedback
					//ModelRuleslst = TrainFalconClassifier(SimilarityList, curr_iter, feedback_val, true);
				//	if (useClassifier)
				//		ooutput = TrainClassifierPython(SimilarityList, curr_iter, feedback_val, true);
					//ModelRuleslst = TrainClassifier(SimilarityList, curr_iter, feedback_val, false);
				//	if(ooutput)
				//		return;
					break;
				}
				if(i==SimilarityList.size())
					end=true;
				/*if(output[0] &&!output[1]&& i<10000){
					pair p = new pair(u,v);
					System.out.println(u+" "+v+" "+Extractfeat(p,output[0]));
					System.out.println(recordList1.get(u)+" "+recordList1.get(v)+" "+Extractfeat(p,output[0]));

				}*/			
			}
			//send validation here
			recallprint.println(curr_iter+" "+gr*2.0/goldMap.keySet().size()+" "+trgr+" "+gr+" "+actualqueried_edge_map.keySet().size()+" "+actual_comparisons);	
			System.out.println(gr+" "+curr_iter+" "+goldMap.keySet().size()+" "+actualqueried_edge_map.keySet().size()+" "+actual_comparisons);	
			if (end)
				break;
		}
				final long endTime = System.currentTimeMillis();

		int iter=0;
		int new_green=0,gtgreen=0;
		for(pair p:actualqueried_edge_map.keySet()){
			System.out.println(p.x+" "+p.y+" "+recordList1.get(p.x)+" "+recordList1.get(p.y));
                                System.out.println(Extractfeat(p,goldMap.containsKey(p)));
		}

		PrintStream out1 = new PrintStream("Blockingeddges.txt");
		for(Integer u:blocking_edges.keySet()){
			out1.println(u+" "+blocking_edges.get(u));
		}
		System.out.println("Total execution time: " + (endTime - startTime) );


		PrintStream pr  = new PrintStream("notcaught.txt");
		for(pair p : goldMap.keySet()) {
			boolean[] out = query_edge_prob( oracle, goldMap, p.x, p.y,0,0);
			if(out[1])
				new_green++;
			if(out[0])
				gtgreen++;
			/*if(!edge_prob.containsKey(p)) {
				iter++;
			}else {
			if(!out[1]){
			*/
				if(!edge_prob.containsKey(p)){
					pr.println(Weightedjaccard(p.x,p.y)+" "+p.x+" "+p.y+" "+recordList1.get(p.x)+" "+recordList1.get(p.y));
					pr.println(inverted_list1.get(p.x)+" "+inverted_list1.get(p.y));
					for(String s: inverted_list1.get(p.x)){
						pr.println(s+" "+blockWeight1.get(s)+" "+blockSize.get(s));
					}for(String s: inverted_list1.get(p.y)){
						pr.println(s+" "+blockWeight1.get(s)+" "+blockSize.get(s));
					}
				

				}//System.out.println(Extractfeat(p,true));
			//}}	

		/*	if(clust_map.containsKey(p.x) &&  clust_map.containsKey(p.y)) {
					if(clust_map.get(p.x).equals(clust_map.get(p.y)))
						System.out.println(p.x+" "+p.y);
				}
		*/	
		}
		System.out.println(iter+" "+edge_prob.keySet().size());
		this.recallprint.println(processed_nodes.size()+" "+this.N+" "+new_green+" "+gtgreen);
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
		BlossCleanClassifierBlockingsongs FF = new BlossCleanClassifierBlockingsongs();

                String line;





		System.out.println("Starting Memory KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
		String[] attributeList ;
//		String line;
		int val;
		for(int i=1;i<=2;i++) {
			String recordFile     = FF.folder+"/records"+Integer.toString(i)+".txt";
			Scanner	scanner = new Scanner(new File(recordFile));
			
			HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
			
			while(scanner.hasNextLine()){
				line= scanner.nextLine();
				line = line.toLowerCase( );
				line=line.replace("!", "");
				line = line.replaceAll("( )+", " ");

				//System.out.println(line);
				attributeList = line.split(";",0);
				//ArrayList<String> record = new ArrayList<String>();
				HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
				int recordId = Integer.parseInt(attributeList[0].trim().replace(".jpg", ""))-1;
				//if(FF.folder=="cora")
				//	recordId++;
				//if(!largest.contains(recordId))
					String[] tokenList = attributeList;
					for(int j=1;j<tokenList.length;j++) {
						//if(tokenList[j].length()<=1)
						//	continue;
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
			pair p2=new pair(id,u);
			FF.goldMap.put(p2, 1);
		}
		
		
		
		System.out.println("Memory after goldMap:" + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		System.out.println("Going inside process");
		System.out.println("Going inside process");
		FF.processBlocks();
	/*	
		PrintStream queried = new PrintStream(FF.folder+"/code_queried.txt");
		//PrintStream queried = new PrintStream(FF.folder+"/goldgreen.txt");
		ArrayList<String> featureOrder = new ArrayList<String>();
		for (pair p:FF.actualqueried_edge_map.keySet()){
			HashMap<String,Double>  dict= FF.Extractfeat(p,FF.actualqueried_edge_map.get(p));
                        if(featureOrder.size()==0){
                                featureOrder.addAll(dict.keySet());
                                queried.println(featureOrder);
                        }
			for (String s: dict.keySet()){
                                if(s.equals("Class"))
                                        continue;
                                else
                                	queried.print(dict.get(s)+",");
                        }
                        queried.print(dict.get("Class")+"\n");


		}*/

	}

}

