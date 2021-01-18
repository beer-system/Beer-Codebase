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


public class BlossDirtyWithoutClassifierCora {
	Random rand = new Random(1991);
	static Random generator = new Random(1992);
	String folder = "cora";//CarsCombined";
	PrintStream recallprint;
	Classifier cModel;
	int num_blocks=0;
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
	String pipelinetype = "pM3loop";//"pM3loop";
	double g_edges = 65000;//343524;//87512;//328411//65000;
	double g_to_r=0.1;
	double r_to_g=0.1;
	boolean useClassifier=false;//true;//false;
	int mergedBlocks=-1;
	boolean classifierExists=false;
	FastVector fvWekaAttributes;
	int featNumbers= 9;
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();

	HashMap<String, ArrayList<Integer>> blockQueries = new HashMap<String,ArrayList<Integer>>();
	HashMap<String, Integer> PrevQueries = new HashMap<String,Integer>();

	HashMap<String,Boolean> updated_blocks = new HashMap<String,Boolean>();

	Set<Integer> processed_nodes = new HashSet<Integer>();
	HashMap<Integer,Integer> nodeClusterMap = new HashMap<Integer,Integer>();	
	HashMap<pair,Boolean> ClustQueryMap = new HashMap<pair,Boolean>();	
	Map<pair, Boolean> inferred_edges = new HashMap<pair, Boolean>();

	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	Map<Integer, ArrayList<String>> inverted_list = new HashMap<Integer, ArrayList<String>>();
	Instances TrainingSet;
	HashMap<String,Boolean> dict = new HashMap<String,Boolean>();

	HashMap<Integer,String> recordList = new HashMap<Integer, String>();
	HashMap<String,Double> blockWeight = new HashMap<String, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	Map<pair, Double> prev_edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	ArrayList<double[]> newSimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();

	ArrayList<Integer> NewNodes = new ArrayList<Integer>();
	HashMap<String,Integer> numGreen = new HashMap<String, Integer>();
	HashMap<String,Integer> numRed = new HashMap<String, Integer>();

	HashMap<String,Integer> blockIndex = new HashMap<String, Integer>();
	HashMap<Integer,String> Index2Str = new HashMap<Integer, String>();
	ArrayList<ArrayList<Integer>> blockList = new ArrayList<ArrayList<Integer>>();
	ArrayList<double[]> blockSize = new ArrayList<double[]>();
	
	HashMap<String,Integer> ruleBlockSize = new HashMap<String,Integer>();
	HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

	
	HashMap<Integer,Integer> goldMap = new HashMap<Integer,Integer>();

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
	public double Weightedjaccardbw (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		
		for(String a:s1.keySet()){
			if(!blockWeight.containsKey(a)) {
				continue;
			}
			//System.out.println(a+" "+union+" "+inter);
			if(s2.containsKey(a)){
				inter+=(blockWeight.get(a)*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight.get(a)*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight.get(a)*s1.get(a));
		}
		for(String a:s2.keySet()){
			//System.out.println(a+" "+union+" "+inter);
			if(!blockWeight.containsKey(a))
				continue;
			
			if(s1.containsKey(a))
				continue;
			else
				union+=(blockWeight.get(a)*s2.get(a));	
		}

		return inter*1.0/union;
	}
	
	ArrayList<Integer> goodcount = new ArrayList<Integer>();
	ArrayList<Integer> badcount = new ArrayList<Integer>();
	public double Weightedjaccardold (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
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
				union+=(s2.get(a));	
		}

		return inter*1.0/union;
	}
	public double Weightedjaccard (int u, int v){
		double inter = 0, union = 0;
		ArrayList<String> l1 =  inverted_list.get(u);
		ArrayList<String> l2 =  inverted_list.get(v);
		
		for(String a:l1){
			if(!blockWeight.containsKey(a))
				continue;
			//System.out.println(a+" "+u);
			if(l2.contains(a)){
				inter+=(blockWeight.get(a));//*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight.get(a));//*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight.get(a));//*s1.get(a));
		}
		for(String a:l2){
			if(!blockWeight.containsKey(a))
				continue;
			if(l1.contains(a))
				continue;
			else
				union+=(blockWeight.get(a));//*s2.get(a));	
		}
		
		if (union==0.0)
			return 0.0;
		return inter*1.0/(union);
	}

	
	 public boolean  TrainClassifierPython(ArrayList<double[]> SimList, int start,int num, boolean newsamples) throws Exception{
         PrintStream classifier_output = new PrintStream(folder+"/classifier.txt");
         PrintStream known_output = new PrintStream(folder+"/known.txt");
         PrintStream unseen_output = new PrintStream(folder+"/unseen.txt");
	if(prev_edge_prob.keySet().size() == edge_prob.keySet().size())
		return true;
         for(int i=start;;i++) {
        	  {
                  if(i>=SimList.size())
                          break;
                  HashMap<String,Double> feat = new HashMap<String,Double>();
                  double[] curr = SimList.get(i);
                  pair p = new pair((int)curr[1],(int)curr[2]);
                  pair p1 = new pair((int)curr[2],(int)curr[1]);
                  if(!edge_prob.containsKey(p))
                          continue;
                  if(goldMap.get(p.x).equals(goldMap.get(p.y)))//containsKey(p))
                          feat=(Extractfeat(p, true));
                  else
                          feat=(Extractfeat(p, false));
                 // if(feat.keySet().size()<15)
                   //       System.out.println("something is weong"+feat+" "+feat.keySet());
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
		if(prev_edge_prob.containsKey(p) || prev_edge_prob.containsKey(p1))
			continue;
		else{
			prev_edge_prob.put(p,0.0);
			prev_edge_prob.put(p1,0.0);
		}
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
                
                 HashMap<String,Double> feat = new HashMap<String,Double>();
                 if(goldMap.get(p.x).equals(goldMap.get(p.y)))
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
         String[] args1 = new String[] {"/bin/bash", "-c", "cd ~/javaCode/src/cora; python learner.py > o1; cd .."};
         Process proc = new ProcessBuilder(args1).start();
          proc.waitFor();
         String pythonoutput = folder+"/inferred.txt";
         String pythonqueried = folder+"/queried.txt";
         String newoutput = folder+"/newoutput.txt";
         Scanner scanner = new Scanner(new File(pythonoutput));

         String line;
          while(scanner.hasNextLine()){
                line= scanner.nextLine();
                 String[] idlist = line.split(" ",0);
                 pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
                 inferred_edges.put(p,true);
         //        System.out.println("inferred"+p.x+" "+p.y);
         }

         scanner = new Scanner(new File(newoutput));
          while(scanner.hasNextLine()){
                line= scanner.nextLine();
                 String[] idlist = line.split(" ",0);
                 pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
                 inferred_edges.put(p,true);
          }

          scanner = new Scanner(new File(pythonqueried));
           while(scanner.hasNextLine()){
                 line= scanner.nextLine();
                  String[] idlist = line.split(" ",0);
                  pair p = new pair(Integer.parseInt(idlist[0]),Integer.parseInt(idlist[1]));
                  if(idlist[2].equals("true")){
                          actualqueried_edge_map.put(p,true);
                  }else{
                          actualqueried_edge_map.put(p,false);
                  }
           //       System.out.println("queried"+p.x+" "+p.y);
          }

	System.out.println("coming back");
          return true;
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
	public boolean[] query_edge_prob(HashMap<pair,Boolean>  oracle, HashMap<Integer, Integer> gt, int u, int v, double r_to_g, double g_to_r) throws Exception{
		boolean ret[] = {false, false};
		pair tmp = new pair(u,v);
		if (gt.get(u).equals(gt.get(v))){
			ret[0]=true;
		}
		pair tmp1 = new pair(tmp.y,tmp.x);
		
		if(useClassifier) {
			if (inferred_edges.containsKey(tmp)|| inferred_edges.containsKey(tmp1))
				ret[1]=true;
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
		ArrayList<String> s1 = inverted_list.get(n1);
		ArrayList<String> s2 = inverted_list.get(n2);
		for(String s:s1){
			if(s2.contains(s)){
				int val = 0;
				if(numGreen.containsKey(s))
					val = numGreen.get(s);
				numGreen.put(s, val+1);
				if (numGreen.get(s)>5000)//100 choose 2
					continue;
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
		ArrayList<String> s1 = inverted_list.get(n1);
		ArrayList<String> s2 = inverted_list.get(n2);
		for(String s:s1){
			if(s2.contains(s)){
				int val = 0;
				if(numRed.containsKey(s))
					val = numRed.get(s);
				numRed.put(s, val+1);
				if (numRed.get(s)>5000)//100 choose 2
					continue;
				
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
			//HashMap<String,Integer> s1 = GetDict(recordList.get(a));
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
			for(String s: inverted_list.get(a)){
					if(PrevQueries.containsKey(s))
					if(PrevQueries.get(s)>100)
                                                        continue;//if(numGreen.containsKey(s) )
                                ArrayList<Integer> lst1;
                               	if(blockQueries.containsKey(s))
                               		lst1 = blockQueries.get(s);
                        	else
                                        lst1 = new ArrayList<Integer>();

                                if (!lst1.contains(a))
                                        lst1.add(a);

				int level = s.split("!",0).length-1;
                                ArrayList<Integer> lst = BlockTreeMap.get(level).get(s);
				//for (int a:lst)
				{
					for(int b:lst){
						if(a!=b){
							if(!nodeClusterMap.containsKey(b))
								continue;
							if(nodeClusterMap.get(a) == nodeClusterMap.get(b)){
								int val = 0;
                           					     if(numGreen.containsKey(s))
                   					                     val = numGreen.get(s);
                               					 numGreen.put(s, val+1);
                                				if (!lst1.contains(b))
                                        				lst1.add(b);
							}else{
								pair p = new pair(nodeClusterMap.get(a),nodeClusterMap.get(b));
								if (ClustQueryMap.containsKey(p)){
								int val = 0;
                                                                     if(numRed.containsKey(s))
                                                                             val = numRed.get(s);
                                                                 numRed.put(s, val+1);
                                                                if (!lst1.contains(b))
                                                                        lst1.add(b);

								}

							}


						}
					}
				}
                                blockQueries.put(s, lst1);						

			}
		}	
/*


			if(!updateneeded)
				continue;
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
	public void UpdateWeightold() throws FileNotFoundException{

	
		PrintStream bw = new PrintStream("bw.txt");
		PrintStream bwand = new PrintStream("and.txt");
		double weightlast=0;
		String slast="";
		for(int i=0;i<blockSize.size();i++) {
			double maxprob = 0.0;
			int queried=0;
			int blockId = (int) blockSize.get(i)[0];
			//System.out.println(blockSize.get(i)[1]+" "+blockId);
			String s = Index2Str.get(blockId);
			ArrayList<Integer> lst  = blockList.get(blockIndex.get(s));
			Collections.shuffle(lst);
			lst.subList(0, Math.min(500,lst.size()));
			
			s=s.replaceAll("/", "");
			PrintStream bw1 = new PrintStream("blocks/"+s);
			
			
			if (i>5185 && !s.equals("chevrolet")) {
				blockWeight.put(s,0.0);//weightlast*Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())/(Math.log(recordList.keySet().size()*1.0/blockList.get(blockIndex.get(slast)).size())*1.0/Math.log(recordList.keySet().size())));

				//blockWeight.put(blockIndex.get(s),weightlast*Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())/(Math.log(recordList.keySet().size()*1.0/blockList.get(blockIndex.get(slast)).size())*1.0/Math.log(recordList.keySet().size())));
				bw.println(i+" "+s+" "+lst.size()+" "+Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())+" "+blockWeight.get(blockIndex.get(s)));
				continue;
			}

			double ng=0, nr=0;
			if(numGreen.containsKey(s))
				ng = numGreen.get(s);
			if(numRed.containsKey(s))
				nr = numRed.get(s);
			queried=(int) (ng+nr);
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
						if(edge_prob.containsKey(t)) {
							if(edge_prob.get(t)>maxprob)
								maxprob = edge_prob.get(t);	
						}
							
						if(goldMap.get(lst.get(i1)).equals(goldMap.get(lst.get(i2))))
							gr++;
						else
							rd++;
						if(edge_prob.containsKey(t) && !goldMap.get(lst.get(i1)).equals(goldMap.get(lst.get(i2))))
							bw1.println( edge_prob.get(t)+" "+lst.get(i1)+" "+lst.get(i2)+" "+goldMap.get(lst.get(i1))+" "+goldMap.get(lst.get(i2)));
					 /*
 *                                                 if(queried_edge_map.containsKey(t)){
 *                                                                                                         //System.out.println("problem ehre");
 *                                                                                                                                                                 if(queried_edge_map.get(t)) {
 *                                                                                                                                                                                                                                 if(clustSize.containsKey(lst.get(i1)))
 *                                                                                                                                                                                                                                                                                                         clustSize.put(lst.get(i1),clustSize.get(lst.get(i1))+1);
 *                                                                                                                                                                                                                                                                                                                                                                         else
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                 clustSize.put(lst.get(i1),1.0);
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 if(clustSize.containsKey(lst.get(i2)))
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         clustSize.put(lst.get(i2),clustSize.get(lst.get(i2))+1);
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         else
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 clustSize.put(lst.get(i2),1.0);
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         }
 *
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 continue;
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 }
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         */ 

						if(nodeClusterMap.containsKey(t.x) && nodeClusterMap.containsKey(t.y)){
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
							for(int j=i11+1;j<lst.size();j++) {

								int v = lst.get(j);
								weighted += Weightedjaccardold(GetDict(recordList.get(u)),GetDict(recordList.get(v)));
								normal += jaccard(GetDict(recordList.get(u)),GetDict(recordList.get(v)));
							}
						}
						weighted = weighted*2.0/(lst.size()*(lst.size()-1));
						normal = normal*2.0/(lst.size()*(lst.size()-1));
						//System.out.println("block is "+s+" "+lst.size()+" "+weighted+" "+normal+" "+i);
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
						blockWeight.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if(fulldummy == 0)
						blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy== 2)
						blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy==3) {
						if(queried>10)
							blockWeight.put(s, Math.pow((gr*1.0/(gr+rd)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else
							blockWeight.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					}else if (fulldummy ==4)
						blockWeight.put(s, Math.pow(ng*1.0/(ng+nr),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy==5)
						blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1));
					else if (fulldummy==6) {
						//if(lst.size()>1)
						blockWeight.put(s, (gr*1.0/(gr+rd))*Math.exp(entropy) );//(maxsize*1.0/lst.size())//clustsize.keySet().size()		
					}
					bw.println(i+" "+s+" "+lst.size()+" "+gr*1.0/(gr+rd)+" "+weighted+" "+normal+" "+Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())+" "+blockWeight.get(blockIndex.get(s)));
					if (i==5099) {
						weightlast= blockWeight.get(blockIndex.get(s));
						slast=s;
					}


					//if(lst.size()>200)
					//blockWeight.put(blockIndex.get(s), (gr*1.0/(gr+rd)));
					//else
					//	blockWeight.put(blockIndex.get(s), Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size()));

				}
				bw1.close();

			}

			//if(ng+nr > 10)
			//System.out.println(ng+" "+nr+" "+blockWeight.get(blockIndex.get(s))+" "+set_clusters.size()+" "+blockList.get(blockIndex.get(s)).size());
		}
	}
	
	public boolean UpdateWeight() throws FileNotFoundException{
		boolean updated = false;
		updated_blocks.clear();
		PrintStream bw = new PrintStream("bw.txt");
		PrintStream bwand = new PrintStream("and.txt");
		for (int i=0;i<BlockTreeMap.keySet().size();i++) {
			HashMap<String, ArrayList<Integer>> level = BlockTreeMap.get(i);
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
				updated_blocks.put(s,true);
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
									normal += jaccard(GetDict(recordList.get(u)),GetDict(recordList.get(v)));
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
							blockWeight.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if(fulldummy == 0)
							blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy== 2)
							blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy==3) {
							if(queried>10)
								blockWeight.put(s, Math.pow((gr*1.0/(gr+rd)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
							else
								blockWeight.put(s, Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						}else if (fulldummy ==4)
							blockWeight.put(s, Math.pow(ng*1.0/(ng+nr),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else if (fulldummy==5)
							blockWeight.put(s, Math.pow(gr*1.0/(gr+rd),1));
						else if (fulldummy==6) {
							//if(lst.size()>1)
							blockWeight.put(s, (gr*1.0/(gr+rd))*Math.exp(entropy) );//(maxsize*1.0/lst.size())//clustsize.keySet().size()		
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
	
	public void GetExpectedOld() throws FileNotFoundException{
		System.out.println(queried_edge_map.keySet().size()+"size is ");
		SimilarityList.clear();
		edge_prob.clear();
		PrintStream Edgeweight = new PrintStream("chevyweight.txt");
		expectedSize.clear();
		
		for(int i=0;i<blockSize.size();i++) {
			System.out.println( (int) blockSize.get(i)[0]+" "+  Index2Str.get((int) blockSize.get(i)[0])+" "+blockList.get(blockIndex.get(Index2Str.get((int) blockSize.get(i)[0]))).size());
		}
		
		System.out.println("DONNNNNNNNNNNNNNNNNNNNNNNNN");
		
		
		PrintStream pt = new PrintStream("blockweight.txt");
		HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
		for(int i=0;i<blockSize.size()-numIgnore;i++) {
			System.out.println(i+" "+blockSize.size());
			
			int blockId = (int) blockSize.get(i)[0];
			//System.out.println(blockSize.get(i)[1]+" "+blockId);
			String s = Index2Str.get(blockId);
			ArrayList<Integer> lst = blockList.get(blockIndex.get(s));
			if (i>5000 && !s.equals("chevrolet"))
				continue;
			
			double gr=0,rd=0;
			for(int i1=0;i1<lst.size();i1++){
				int u = lst.get(i1);
				for(int i2=i1+1;i2<lst.size();i2++){
					int v = lst.get(i2);
					if(goldMap.get(u).equals(goldMap.get(v)))
						gr++;
					else rd++;
					pair t = new pair(u,v);
					pair t1 = new pair(v,u);
					if(processed.containsKey(t) || processed.containsKey(t1))
						continue;
					else {
						double[] p = {Weightedjaccardold(GetDict(recordList.get(u)),GetDict(recordList.get(v))), u, v};
						if (goldMap.get(u)==54 && goldMap.get(v)==54) {
							Edgeweight.println(u+" "+v+" "+p[0]);
						}
						SimilarityList.add(p);
						if(u==994 && (v==1812))
							System.out.println(p[0]+" "+recordList.get(u)+" "+recordList.get(v));
						//if (u==38573 && v ==70489)
						//System.out.println(recordList.get(u)+" "+recordList.get(v)+" "+p[0]);
						processed.put(t1, true);
						processed.put(t, true);
					}
				}
			}

			if(lst.size()>1)
				pt.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd));
		}
		pt.close();
		System.out.println("similarity list size is "+SimilarityList.size());
		System.out.println(queried_edge_map.keySet().size()+"size is ");



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
		System.out.println("maximum similarity is "+maxsim+" "+goldMap.get((int)SimilarityList.get(0)[1])+" "+goldMap.get((int)SimilarityList.get(0)[2]));
		ArrayList<double[]> bucket =new ArrayList<double[]>();
		double max = 1.0,min = 0.99;
		double gr =0;
		double rd = 0;
		PrintStream probvales = new PrintStream("probvalues.txt");
		expectedSize.clear();
		double numEdges = 0,prevgr=0,prevrd=0;
		System.out.println(queried_edge_map.keySet().size()+"size is ");
		newSimilarityList.clear();
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
						pair t1 = new pair(u,v);
						pair t2 = new pair(v,u);
		
						if(!edge_prob.containsKey(t1)){
							newSimilarityList.add(p);

						}
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
						probvales.println(u+" "+v+" "+p[0]+" "+pr+" "+goldMap.get(u).equals(goldMap.get(v)));
					}
					if(numEdges>100) {
						prevgr=gr;
						prevrd=rd;
					}
					gr=0;
					rd=0;
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
				double p;
				if(queried_edge_map.containsKey(t) && !pipelinetype.equals("ploop")){
					if(queried_edge_map.get(t))
						p=1.0;
					else
						p=0;
					numEdges++;
				}else {
					if(goldMap.get(t.x).equals(goldMap.get(t.y)))
						p=1.0;
					else
						p=0;
				//	p= curr[0];
				}
				//if(u==994 && (v==1812))
				//if(curr[1]==994 && curr[2]==1812)
				gr+=p;
				rd+=(1.0-p);
				//if(min==0.009999999999999247)
				//	System.out.println(p+"DDDDDD"+gr+ " "+rd+" "+i+" "+queried_edge_map.keySet().size()+" "+bucket.size()+" "+min);
				
			}


		}





		//SimilarityList.clear();	
	}
			
	public void GetExpected(boolean first) throws FileNotFoundException{
		//System.out.println(queried_edge_map.keySet().size()+"size is ");
		SimilarityList.clear();
		//edge_prob.clear();
		expectedSize.clear();		
		//USE processed_nodes here and partition similarity in buckets so it is easy to store vslues for each bucket
		//
        	 HashMap<Integer,Integer> node_degree = new HashMap<Integer,Integer>();
		
			HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
			Comparator<String> comparator = new ValueComparator(blockWeight);
			
			TreeMap<String, Double> result = new TreeMap<String, Double>(comparator);
	        	result.putAll(blockWeight);
        	
			int iter=0;
			boolean foundnew=false;
			for (String key : result.keySet()) { 
				
				String s = key;
				//if(!updated_blocks.containsKey(s))
			//		continue;	
				System.out.println(s+" "+SimilarityList.size());
				int level = s.split("!",0).length-1;
				ArrayList<Integer> lst = BlockTreeMap.get(level).get(s);
				//System.out.println(iter+" "+key+" "+lst.size()+" "+blockWeight.get(key)+" "+processed.keySet().size());
				iter++;
				if(SimilarityList.size()>10000000)
					break;
		
				//if (iter >= 0.99*result.keySet().size())
				//if (iter >= result.keySet().size()-20 && !s.equals("chevrolet"))
				//	continue;	
				//if (lst.size()>3000 && !s.equals("chevrolet"))
				//	continue;
				
				for(int i1=0;i1<lst.size();i1++){
					int u = lst.get(i1);
					for(int i2=i1+1;i2<lst.size();i2++){
						int v = lst.get(i2);
						pair t = new pair(u,v);
						pair t1 = new pair(v,u);
						if(processed.containsKey(t) || processed.containsKey(t1))
							continue;
						else {
							double[] p = {Weightedjaccard(u,v), u, v};
							SimilarityList.add(p);
							processed.put(t1, true);
							processed.put(t, true);
							//if(first)
							{
								double val1=0.0,val2=0.0;
								if( expectedSize.containsKey(u))
                                                                	val1=expectedSize.get(u);
                                                        	if( expectedSize.containsKey(v))
                                                                	val2=expectedSize.get(v);
								val1+=p[0];
								val2+=p[0];
                                                        	expectedSize.put(u, val1);
                                                        	expectedSize.put(v, val2);
						
								if(node_degree.containsKey(u))
		                                                        node_degree.put(u,node_degree.get(u)+1);
                		                                else
                                		                        node_degree.put(u,1);
                                                		if(node_degree.containsKey(v))
                                           		             node_degree.put(v,node_degree.get(v)+1);
                                                		else
                                                        		node_degree.put(v,1);
								edge_prob.put(t,p[0]);
								edge_prob.put(t1,p[0]);

							}
							//edge_prob.put(t1,p[0]);
							//edge_prob.put(t,p[0]);
						}
					}
	        		}
        	
			}

			processed.clear();
			//System.out.println("similarity list size is "+SimilarityList.size());
	        	//System.out.println(queried_edge_map.keySet().size()+"size is ");
			//if(first)
			//	return;        	
			//if(!foundnew)
	        	//	return;
        	
			//This is probabilit conversion of sim

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
                       //         if(blocking_edges.get(p.x)>100 || blocking_edges.get(p.y)>100){
                       
                         //                           }


                        double[] cfibf = get_cficf_score(p);
                        int loc = (int)(cfibf[2]*1.0/blossT);
                        ArrayList<pair> tmp = levelList.get(loc);
                        ArrayList<Integer> feattmp = new ArrayList<Integer>();
                        feattmp.add(Math.min(1000000,node_degree.get(p.x)));
                        feattmp.add(Math.min(100000,node_degree.get(p.y)));
                        feattmp.add((int)cfibf[0]);
                        feattmp.add((int)cfibf[1]*100);
                        feattmp.add((int)cfibf[2]*100);
                        blocking_features.put(p,feattmp);
                        tmp.add(p);
                        levelList.set(loc,tmp);

                }
                System.out.println(blocking_features.keySet().size());
                process_pairs_levels(levelList,blocking_features);

			Collections.sort(SimilarityList, new Comparator<double[]>() {
				public int compare(double[] o1, double[] o2) {
					double s1 = o1[0]; double s2 = o2[0];
					if (s1 != s2)
						return (s1 > s2 ? -1 : 1);
					else
						return 0;
				}
			});
			double maxsim = SimilarityList.get(0)[0];
			ArrayList<double[]> bucket =new ArrayList<double[]>();
			double max = 1.0,min = 0.99;
			double gr =0;
			double rd = 0;
			double gtgr=0,gtrd=0;
			PrintStream probvales = new PrintStream("probvalues.txt");
			expectedSize.clear();
			double numEdges = 0,prevgr=0,prevrd=0;
	        	System.out.println(queried_edge_map.keySet().size()+"size is ");
        		edge_prob.clear();
			newSimilarityList.clear();
			for(int i=0 ;i<SimilarityList.size();i++){
				double[] curr = SimilarityList.get(i) ;
				if(curr[0] > 1.0)
					curr[0]=1.0;
				if(curr[0]<min || i==SimilarityList.size()-1){
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
							if(!edge_prob.containsKey(t1)){
								newSimilarityList.add(p);

							}
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
                int numpairperlevel=50;
                ArrayList<pair> final_candidates = new ArrayList<pair>();
                for (ArrayList<pair> level: levelList){
                        Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
                        Collections.shuffle(level,newgen);
                        final_candidates.addAll(level.subList(0, Math.min(50,level.size())));

                }

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
                int higher = 0;
                 PrintStream pt = new PrintStream(folder+"/Blosstrain.txt");
                PrintStream blossfalcon = new PrintStream(folder+"/Blossfalcon.txt");
                int it=0;
                for(pair p : training){
                        ArrayList<Integer> feat = blocking_features.get(p);
                        if(it==0) pt.println("a,b,c,d,e,f,id1,id2,Class");
                        it++;
			for (int f:feat){
                                pt.print(f+",");
                        }
                        if(goldMap.get(p.x).equals(goldMap.get(p.y)))
                                pt.print(edge_prob.get(p)+","+p.x+","+p.y+",1.0\n");
                        else    pt.print(edge_prob.get(p)+","+p.x+","+p.y+",0.0\n");


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
                        if(goldMap.get(p.x).equals(goldMap.get(p.y)))
                                btest.print(edge_prob.get(p)+","+p.x+","+p.y+",1.0\n");
                        else    btest.print(edge_prob.get(p)+","+p.x+","+p.y+",0.0\n");
                }
                btest.close();





                String[] args1 = new String[] {"/bin/bash", "-c", "cd ~/javaCode/src/cora; python learnerbloss.py > b1; cd .."};
                
		 try{
                Process proc = new ProcessBuilder(args1).start();
                proc.waitFor();
                }catch(Exception e){
                        ;
                }
                String blossoutput = folder+"/blossoutput.txt";
                Scanner scanner = new Scanner(new File(blossoutput));
                SimilarityList.clear();
                while(scanner.hasNextLine()){
                        String line= scanner.nextLine();
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
                int inter=0,num1=0,num2=0;
                ArrayList<String> list2 = inverted_list.get(p.y);
                for (String s1: inverted_list.get(p.x)){
                   int level = s1.split("!",0).length-1;
                                if(level>0)
                                        continue;
                                num1++;
                        if(list2.contains(s1)){
                                inter++;
                                reciprocal_score+=1.0/(BlockTreeMap.get(level).get(s1).size() + BlockTreeMap.get(level).get(s1).size());
                        }
                }
                for(String s2:list2){
                        int level = s2.split("!",0).length-1;
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
	public HashMap<pair,Boolean> print_edges(HashMap<Integer, Integer> gt){

		HashMap<pair,Boolean> queried_edge_map = new HashMap<pair,Boolean>();

		int n1 = 0,n2 = 0;
		try{
			//System.out.println("AAAA");
			while(n1<recordList.keySet().size()){
				n2=n1+1;
				while(n2<recordList.keySet().size()){
					pair tmp1 = new pair(n1,n2);
					pair tmp2 = new pair(n2,n1);
					if (gt.get(n1).equals(gt.get(n2))){
						{
							double rand1 = this.rand.nextDouble();
							if( rand1 >= g_to_r){
								queried_edge_map.put(tmp1,true);
								queried_edge_map.put(tmp2,true);

								//System.out.println("printing");

							}
							else{
								//System.out.println("false" + rand);

								queried_edge_map.put(tmp1,false);
								queried_edge_map.put(tmp2,false);

							}
						}

					}else{
						{
							if(this.rand.nextDouble() <= r_to_g){
								queried_edge_map.put(tmp1,true);
								queried_edge_map.put(tmp2,true);

							}
							else{
								queried_edge_map.put(tmp1,false);
								queried_edge_map.put(tmp2,false);

							}
						}


					}
					n2++;
				}
				n1++;
			}




		} catch (Exception e) {
			System.out.println("Exception raised");
		}
		return queried_edge_map;
	}

	public HashMap<String,Double> extractFeatures(pair p){
		HashMap<String,Double> featureVect = new HashMap<String,Double>();
		String rec1 = recordList.get(p.x);
		String rec2 = recordList.get(p.y);
		HashMap<String,Integer> recmap1 = GetDict(rec1);
		HashMap<String,Integer> recmap2 = GetDict(rec2);


		featureVect.put("jaccard", jaccard(recmap1,recmap2));
		featureVect.put("Overlap", Overlap(recmap1,recmap2));
		featureVect.put("weightedJaccard", Weightedjaccardold(recmap1,recmap2));
		featureVect.put("Cosine", CosineSim(recmap1,recmap2));
		return featureVect;
	}
	public Double Test(int u,int v, Classifier C) throws Exception {

		pair p = new pair(u,v);

		// System.out.println(p.x+" "+p.y+" "+features.get("jaccard")+" "+features.get("weightedJaccard")+" "+features.get("Cosine"));
		ArrayList<String> Dictionary = new ArrayList<String>(dict.keySet());

		int featNumbers=5+ dict.keySet().size();
		Attribute feature1 = new Attribute("jaccard");
		Attribute feature2 = new Attribute("weightedJaccard");
		Attribute feature3 = new Attribute("Cosine");
		Attribute feature4 = new Attribute("Overlap");

		ArrayList<Attribute> features = new ArrayList<Attribute>();
		for(String s:Dictionary) {
			Attribute tmp = new Attribute(s);
			features.add(tmp);
		}
		// Declare the class attribute along with its values
		FastVector classLabel = new FastVector(2);
		classLabel.addElement("green");
		classLabel.addElement("red");
		Attribute ClassAttribute = new Attribute("Class", classLabel);

		// Declare the feature vector
		FastVector fvWekaAttributes = new FastVector(featNumbers);
		fvWekaAttributes.addElement(feature1);
		fvWekaAttributes.addElement(feature2);
		fvWekaAttributes.addElement(feature3);
		fvWekaAttributes.addElement(feature4);
		for(Attribute s:features) {
			fvWekaAttributes.addElement(s);
		}
		fvWekaAttributes.addElement(ClassAttribute);

		Instances TrainingSet2 = new Instances("Rel", fvWekaAttributes, 10);
		// Set class index
		TrainingSet2.setClassIndex(featNumbers-1);


		Instance curr = new DenseInstance(featNumbers);

		//System.out.println(features.size()+" "+Dictionary.size());
		HashMap<String,Double> Simfeatures = extractFeatures(p);
		ArrayList<String> lst = Intersect(recordList.get(p.x), recordList.get(p.y));
		//System.out.println(p.x+" "+p.y+" "+features.get("jaccard")+" "+features.get("weightedJaccard")+" "+features.get("Cosine")+" "+trainingSet.get(p));
		curr.setValue((Attribute)fvWekaAttributes.elementAt(0), Simfeatures.get("jaccard"));
		curr.setValue((Attribute)fvWekaAttributes.elementAt(1), Simfeatures.get("weightedJaccard"));
		curr.setValue((Attribute)fvWekaAttributes.elementAt(2), Simfeatures.get("Cosine"));
		curr.setValue((Attribute)fvWekaAttributes.elementAt(3), Simfeatures.get("Overlap"));
		for(int i=4;i<features.size()+4;i++) {
			if(lst.contains(Dictionary.get(i-4)))
				curr.setValue((Attribute)fvWekaAttributes.elementAt(i), 1.0);
			else
				curr.setValue((Attribute)fvWekaAttributes.elementAt(i), 0.0);
		}



		curr.setDataset(TrainingSet);

		double[] fDistribution = C.distributionForInstance(curr);
		// System.out.println(fDistribution[0]+" "+fDistribution[1]);
		return fDistribution[0];
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
		featMap.put("jacc", jacc_word[0]);
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
	public HashMap<String,Double> Extractfeat(pair p, Boolean cl){
		
		String[] s1 = (recordList.get(p.x)).split(";",0);
		String[] s2 = recordList.get(p.y).split(";",0);
		ArrayList<Double> featList = new ArrayList<Double>();
		
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		String rec1 = "1 ",rec2="2 ";	
		//System.out.println(p.x+" "+p.y);
		for (int i=1;i<8;i++) {
			if(s1.length<=i || s2.length<=i) {
				featList.add(0.0);
				HashMap<String,Double> CurrFeatMap;
				//if (i==1 || i==2) 
				{
					CurrFeatMap  = Get_small_string_features("","");
					//System.out.println(i+" "+TitleFeatMap);
				}
				
				for (String s:CurrFeatMap.keySet()) {
					featMap.put(Integer.toString(i)+s,CurrFeatMap.get(s) );
				}
			}else {
				if(s1[i].equals("") || s2[i].equals("")) {
					featList.add(0.0);
				}else{
					rec1+= s1[i]+" ";
					rec2+= s2[i]+" ";
					//System.out.println(i+"and"+s1[i]+"and"+s2[i]);
					Double val = Weightedjaccardold(GetDict("1 "+s1[i]),GetDict("2 "+s2[i]));
					if(val.isNaN())
						val=0.0;
					if (i<5)
					featList.add(val);
					else
					featList.add(Weightedjaccardbw(GetDict("1 "+s1[i]),GetDict("2 "+s2[i])));//s1[i],s2[i]));
				}
				
			
			
//			System.out.println(recordList1.get(p.x)+" "+s1);
			HashMap<String,Double> CurrFeatMap;
			//if (i==1 || i==2) 
			{
				CurrFeatMap  = Get_small_string_features(s1[i],s2[i]);
				//System.out.println(i+" "+TitleFeatMap);
			}
			
			for (String s:CurrFeatMap.keySet()) {
				featMap.put(Integer.toString(i)+s,CurrFeatMap.get(s) );
			}
			}
		}
	 		//featMap.put("OverallJacc",Weightedjaccardold(GetDict(rec1),GetDict(rec2)));
	 		featMap.put("title", (featList.get(0)));
	 		featMap.put("authors", (featList.get(1)));
	 		featMap.put("year", (featList.get(2)));
	 		featMap.put("journal", (featList.get(3)));
	 		featMap.put("volume",Weightedjaccard(p.x,p.y));
	 		//featMap.put("volume", (featList.get(4)));
	 		featMap.put("pages", (featList.get(4)));
	 		featMap.put("misc", (featList.get(5)));
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
	
	
	public ArrayList<HashMap<String,Double[]>> TrainClassifier(HashMap<pair,Boolean> training) throws Exception {
		
		int trainingSetsize=training.keySet().size();
		 
		 Attribute feature1 = new Attribute("title");
		 Attribute feature2 = new Attribute("authors");
		 Attribute feature3 = new Attribute("year");
		 Attribute feature4 = new Attribute("journal");
		 Attribute feature5 = new Attribute("volume");
		 Attribute feature6 = new Attribute("pages");
		 Attribute feature7 = new Attribute("misc");
		 Attribute feature8 = new Attribute("sim");
		 
		 FastVector classLabel = new FastVector(2);
		 classLabel.addElement("red");
		 classLabel.addElement("green");
		 Attribute ClassAttribute = new Attribute("Class", classLabel);
		 
		 // Declare the feature vector
		  fvWekaAttributes = new FastVector(featNumbers);
		 fvWekaAttributes.addElement(feature1);
		 fvWekaAttributes.addElement(feature2);
		 fvWekaAttributes.addElement(feature3);
		 fvWekaAttributes.addElement(feature4);
		 fvWekaAttributes.addElement(feature5);
		 fvWekaAttributes.addElement(feature6);
		 fvWekaAttributes.addElement(feature7);
		 fvWekaAttributes.addElement(feature8);
		
		
		 fvWekaAttributes.addElement(ClassAttribute);
		 
			// Create an empty training set
		 TrainingSet = new Instances("Rel", fvWekaAttributes, trainingSetsize);
		 // Set class index
		 TrainingSet.setClassIndex(featNumbers-1);
			 
			 // Create the instance
			for(pair p:training.keySet()) {
				 Instance curr = new DenseInstance(featNumbers);
				 HashMap<String,Double> feat = Extractfeat(p,training.get(p));
				 //System.out.println(feat);
				 //ArrayList<String> lst = Intersect(recordList.get(p.x), recordList.get(p.y));
				 //System.out.println(p.x+" "+p.y+" "+features.get("jaccard")+" "+features.get("weightedJaccard")+" "+features.get("Cosine")+" "+trainingSet.get(p));
				 if (feat.get("title") >= 0)
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
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(7), 0.0);//edge_prob.get(p));
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(7), 0.0);
				 
				 if(feat.get("Class")>0)
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "green");
				 else
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(featNumbers-1), "red");
				 TrainingSet.add(curr);
			 }
			 
			
			  
			 Classifier cModel  =new RandomTree();// new J48();//new Logistic();//new J48();
			// cModel.SetSeed((int)generator.nextDouble()*Integer.MAX_VALUE);
			// String[] opt = {"depth 5"};
			 cModel.SetDepth(5);
			
			 
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
	public ArrayList<ArrayList<HashMap<String,Double[]>>> TrainClassifier(ArrayList<double[]> neighSizelst, int start,int num) throws Exception {
		
		/*double numelem = neighSizelst.get(0)[0];
		for (double[] lst: neighSizelst) {
			
		}*/
		
		training = new HashMap<pair,Boolean>();
		
		ArrayList<Integer> highexpectedsize=new ArrayList<Integer>();
		for(int i=start;i<start+num;i++) {
			if(i>=neighSizelst.size())
				break;
			highexpectedsize.add((int)neighSizelst.get(i)[1]);
		}
		
		ArrayList<double[]> problist = new ArrayList<double[]>();
		for(int i=0;i<highexpectedsize.size();i++) {
			for(int j=i+1;j<highexpectedsize.size();j++) {
				pair p = new pair(highexpectedsize.get(i),highexpectedsize.get(j));
				if(!edge_prob.containsKey(p))
					continue;
				double[] tmp = {p.x,p.y,edge_prob.get(p)};
				problist.add(tmp);
			}
		}
		
		Collections.sort(problist, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[2]; double s2 = o2[2];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});
	
		int g=0;
		/*for(int i=0;i<problist.size();i++) {
			if(g>=5)
				break;
			pair p =new pair((int)problist.get(i)[0],(int)problist.get(i)[1]);
			if(goldMap.get((int)problist.get(i)[0]).equals(goldMap.get(((int)problist.get(i)[1])))) {
				training.put(p, true);
				g++;
			}else
				training.put(p, false);
		}
		*/
		
		System.out.println("training1"+training.size());
		ArrayList<Integer>nodes  = new ArrayList<Integer>();
		/*for(pair p:training.keySet()) {
			if(!nodes.contains(p.x))
				nodes.add(p.x);
			
			if(!nodes.contains(p.y))
				nodes.add(p.y);
		}*/
		nodes.addAll(highexpectedsize);
		System.out.println("number ofnodes"+nodes.size());
		ArrayList<double[]> Simlist=new ArrayList<double[]>();
		for(int i: recordList.keySet()) {
			if(nodes.contains(i))
				continue;
			else {
				for(int j:nodes) {
					pair p = new pair(j,i);
					if(!edge_prob.containsKey(p))
							continue;
					double[] tmp = {i,j,edge_prob.get(p)};
					Simlist.add(tmp);
				}
			}
		}
		Collections.shuffle(Simlist,new Random(rand.nextInt()));
		
		Collections.sort(Simlist, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[2]; double s2 = o2[2];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});
		//training.putAll(actualqueried_edge_map);
		double curr = 1.05;
		for(int i=0 ;i<Simlist.size();i++){
			//System.out.println(SimilarityList.get(i)[0]+" "+curr);
			if(Simlist.get(i)[2] < (curr - 0.03)) {
				pair tmp = new pair((int)Simlist.get(i)[0],(int)Simlist.get(i)[1]);
				if(goldMap.get(tmp.x).equals(goldMap.get(tmp.y)))
					training.put(tmp, true);
				else
					training.put(tmp, false);
				actualqueried_edge_map.put(tmp, training.get(tmp));
				curr = Simlist.get(i)[2];
				System.out.println(tmp.x+" "+tmp.y+" "+Simlist.get(i)[2]+" "+goldMap.get(tmp.x).equals(goldMap.get(tmp.y)));
			}
		}
		
		System.out.println("final training"+training.size());
		
		/*
		training = new HashMap<pair,Boolean>();
		double curr = 1.05;
		for(int i=0 ;i<SimilarityList.size();i++){
			//System.out.println(SimilarityList.get(i)[0]+" "+curr);
			if(SimilarityList.get(i)[0] < (curr - 0.03)) {
				pair tmp = new pair((int)SimilarityList.get(i)[1],(int)SimilarityList.get(i)[2]);
				if(goldMap.get(tmp.x).equals(goldMap.get(tmp.y)))
					training.put(tmp, true);
				else
					training.put(tmp, false);
				curr = SimilarityList.get(i)[0];
				System.out.println(tmp.x+" "+tmp.y+" "+SimilarityList.get(i)[0]+" "+goldMap.get(tmp.x).equals(goldMap.get(tmp.y)));
			}
		}
		System.out.println(training.keySet().size()+"size is ");
		
		*/
		int numClassifiers=10;
		int datasetSize=(int)(training.size()*0.6);
		ArrayList<ArrayList<HashMap<String,Double[]>>> rulelstlst = new ArrayList<ArrayList<HashMap<String,Double[]>>>();
		for(int i=0;i<10;i++) {
			HashMap<pair,Boolean>smallTraining = new HashMap<pair,Boolean>();
			ArrayList<pair> keylst=new ArrayList(training.keySet());
			Collections.shuffle(keylst,new Random(rand.nextInt()));
			for(int j=0;j<datasetSize;j++)
				smallTraining.put(keylst.get(j), training.get(keylst.get(j)));
			ArrayList<HashMap<String,Double[]>> rulelst = TrainClassifier(smallTraining);
			rulelstlst.add(rulelst);
		}
		
		//ArrayList<HashMap<String,Double[]>> rulelst = TrainClassifier(training);
		
		/* for(HashMap<String,Double[]> rule:rulelst) {
				 g=0;
				 int r=0;
				ArrayList<pair> pairlst = new ArrayList<pair>();
				for(double[] lst: SimilarityList) {
					
					pair pp = new pair((int)lst[1],(int)lst[2]);
					HashMap<String,Double> feat = Extractfeat(pp,true);
					feat.put("sim", edge_prob.get(pp));
					//System.out.println(feat);
					
					if(satisfyRule(rule,feat)) {
						//System.out.println(feat+"satisfying" );
						
						pairlst.add(pp);
					}
				}
				
				Collections.shuffle(pairlst,new Random(rand.nextInt()));
				g=0;
				r=0;
				HashMap<pair,Boolean> training2  = new HashMap<pair,Boolean>();
				for(int i=0;i<10;i++) {
					pair pp = pairlst.get(i);
					training2.put(pp,goldMap.get(pp.x).equals(goldMap.get(pp.y)));
					Double pred = rule.get("class")[1];
					if(pred > 0 && goldMap.get(pp.x).equals(goldMap.get(pp.y))) {
						g++;
					}
					else if(pred <= 0 && !goldMap.get(pp.x).equals(goldMap.get(pp.y))) {
						g++;
					}else {
						r++;
					}
				}
				System.out.println("Accuracyof rule"+rule+" "+g+" "+r);
				
				if(g*1.0/(g+r) < 0.8) {
					
					for(pair p:training.keySet()) {
						HashMap<String,Double> feat = Extractfeat(p,true);
						feat.put("sim", edge_prob.get(p));
						
						if(satisfyRule(rule,feat))
							training2.put(p, goldMap.get(p.x).equals(goldMap.get(p.y)));
					}
					ArrayList<HashMap<String,Double[]>> rulelst2 =  TrainClassifier(training2);
					
				}
				
				
		 }*/
		return rulelstlst;
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
	public double Test( pair p,Boolean cl) throws Exception {

			HashMap<String,Double> feat = Extractfeat(p,cl);
			feat.put("sim", edge_prob.get(p));
			// System.out.println(ModelRules.size()+"Number of rules ");
			 int iter=0,out=0;
			 for(ArrayList<HashMap<String,Double[]>> Tree: ModelRuleslst) {
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
			 
			 return out*1.0/ModelRuleslst.size();
			//return eTest.evaluateModelOnce(cModel, curr);	 
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
		int depth=10;
		HashMap<Integer,HashMap<String,ArrayList<Integer>>> tmpBlockTreeMap = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();
		
		for (int i=0;i<depth;i++) {
			HashMap<String,ArrayList<Integer>> tmp = new HashMap<String,ArrayList<Integer>>();
			tmpBlockTreeMap.put(i,tmp);
		}
		
		
		for(Integer rec_id : recordList.keySet()) {
			//System.out.println(rec_id);
			if(recordList.get(rec_id).contains("don't") )
				System.out.println("**********found"+" "+recordList.get(rec_id));
						
			String[] partitions = recordList.get(rec_id).split(";",0);
			String tmp = recordList.get(rec_id).replaceAll(";"," ");
                        tmp=tmp.replaceAll("( )+", " ");
                        String[] rec = tmp.split(" ",0);
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
		if(tmpBlockTreeMap.get(0).containsKey("don't"))
			System.out.println("**********found");
		
		int counter=0;
		int loc=0;
		for(int i=0;i<depth;i++) {
			HashMap<String,ArrayList<Integer>> level_map =tmpBlockTreeMap.get(i);
			HashMap<String,ArrayList<Integer>> cleaned_level_map = new HashMap<String,ArrayList<Integer>>();
			//System.out.println("thos"+BlockTree.get(1).keySet().size());//+" "+level+" "+level_map.keySet().size());
			//System.out.println(i+" "+level_map.keySet().size());
			
			for(String s:level_map.keySet()) {
	//			System.out.println(s+" "+level_map.get(s).size());
				String[] tokenLst = s.split("!");
				if(level_map.get(s).size()>10 || tokenLst.length==1) {
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
							tmp.add(s);
							inverted_list.put(a, tmp);
						}
						counter++;
						blockIndex.put(s,loc);
						Index2Str.put(loc, s);
						double[] tmp = {loc,level_map.get(s).size()};
						num_blocks++;
						blockSize.add(tmp);
						loc+=1;
					}
				}
			}
			BlockTreeMap.put(i, cleaned_level_map);
		}
		
		if(BlockTreeMap.get(0).containsKey("don't"))
			System.out.println("**********found");
		
		
		Collections.sort(blockSize, new Comparator<double[]>() {
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
		return 0	;
	}
	
	public void processBlocks() throws Exception{
		if (mergeBlocks()==0) {
			System.out.println("done merging");
		}
			
		
		
		recallprint= new PrintStream(folder+"/recall.txt");
		PrintStream Nodeorder= new PrintStream("nodeorder.txt");
		
//		HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

		System.out.println(blockSize.size()+"DSDDDDDDDD"+recordList.keySet().size());
		int blockId=0;		
		for(int i=0;i<BlockTreeMap.keySet().size();i++) {
			HashMap<String,ArrayList<Integer>> level = BlockTreeMap.get(i);
			for(String s:level.keySet()) {
			//	if (i==0) {
					if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
						blockWeight.put(s, Math.log(recordList.keySet().size()*1.0/level.get(s).size())*1.0/Math.log(recordList.keySet().size()));
					}else {
						blockWeight.put(s, 1.0);
					}	
			//	}else {
			//		blockWeight.put(s, 0.0);

		/*		if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
					blockWeight.put(s, Math.log(recordList.keySet().size()*1.0/level.get(s).size())*1.0/Math.log(recordList.keySet().size()));
				}else {
					blockWeight.put(s, 1.0);
				}*/
			//}	
		}
		}
		
		
		if(blockWeight.keySet().size()>0) {
			System.out.println("Number of blocks "+blockWeight.size());
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
			GetExpected(true);
			System.out.println("edges "+edge_prob.keySet().size());
			//if(edge_prob.keySet().size()>0)
			//	return;
			System.out.println("done expected");	
		//	if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop"))
		//		boolean updated=UpdateWeight();	
			System.out.println("done updateweight");	
		}
		//if(pipelinetype.equals("pM3loop")|| pipelinetype.equals("ploop"))
		//	GetExpected();
		//if(i1>0)
		//	return;
		System.gc();
		System.out.println("process2 KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		
		System.out.println("done expected");	

		ArrayList<double[]> neighSizelst =new ArrayList<double[]>(); 
		for(int id:expectedSize.keySet()){
			double p;
			//ADDED LATER
			if(processed_nodes.contains(id))
				continue;
			p=expectedSize.get(id);
			double[] tmp ={p,id};

			neighSizelst.add(tmp);
		}
		Collections.sort(neighSizelst, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[0]; double s2 = o2[0];
				if (s1 != s2)
					return (s1 > s2 ? -1 : 1);
				else
					return 0;
			}
		});
		System.out.println(edge_prob.keySet().size()+"edgeprob");
		 while(actualqueried_edge_map.keySet().size()<=2) {
                        pair  randomKey = (pair) edge_prob.keySet().toArray()[new Random().nextInt(edge_prob.keySet().size())];
                        if(!goldMap.get(randomKey.x).equals(goldMap.get(randomKey.y))){//containsKey(randomKey)) {
                                actualqueried_edge_map.put(randomKey, false);
                        }
                        if(actualqueried_edge_map.keySet().size()==2)
                                break;
                }
                while(actualqueried_edge_map.keySet().size()<=4) {
                        pair  randomKey = (pair) edge_prob.keySet().toArray()[new Random().nextInt(edge_prob.keySet().toArray().length)];
                        pair  randomKey1 = new pair(randomKey.y,randomKey.x);
			if(goldMap.get(randomKey.x).equals(goldMap.get(randomKey.y))){
                        //if(goldMap.containsKey(randomKey)) {
                                actualqueried_edge_map.put(randomKey, true);
                        }
                        if(actualqueried_edge_map.keySet().size()==4)
                                break;
                }
		boolean ooutput=false;
		if(useClassifier)
	 	ooutput =  TrainClassifierPython( SimilarityList,0,10000000,true);//int num, boolean newsamples) throws Exception{
		System.out.println("returned from python");
		while(true){
			ArrayList<Integer> QueriedClusters = new ArrayList<Integer>();
			int actualCluster = -1;
			int max=-1;
			double max_benefit=-1;
			int maxnode=-1;
			double maxb = -1;
			
			int num1=0;
			for (double[] nodelst : neighSizelst) {
				int     u  = (int) nodelst[1] ;
				max_benefit=nodelst[0];
				
				if(processed_nodes.contains(u))
					continue;
				else{
					max =  u;
					ArrayList<double[]> comp_list = get_benefit_component(max);
					if(comp_list.size()>0) {
						if(comp_list.get(0)[0]>maxb) {
							maxb = comp_list.get(0)[0];
							maxnode=u;
						}
					}else
						maxnode=u;
					//System.out.println("max is "+max+" "+nodelst[0]);
				}
				num1++;
				
				
					break;
				
			}
			//System.out.println(neighSizelst.get(400)[0]+" ssss"+neighSizelst.get(400)[1]);
			max=maxnode;
			Nodeorder.println(max+" "+goldMap.get(max)+" "+max_benefit);
			if(set_clusters.size()==0){
				ArrayList<Integer> nodes = new ArrayList<Integer>();
				nodes.add(max);
				nodeClusterMap.put(max,0);
				component curr_comp = new component(nodes);
				set_clusters.add(curr_comp);
				processed_nodes.add(max);	
				NewNodes.add(max);
				continue;
			}
			if(max==-1) {
				
				System.out.println("exiting"+neighSizelst.size());// || queries > 2000)
				break;
			}
				
				
			//Write a function to return the benefit between the pair of clusters in processed if they have a queryleft

			ArrayList<double[]> comp_list = get_benefit_component(max);

			int num=0;
			boolean added = false;
			boolean less=false;
			for(double[] comp : comp_list){


				//Maintain a mapping of classifier for each cluster
				//Maintain a different set of asked questions so that we can use it if needed
				//Two options: A random forest for complete dataset or a classifier per cluster...
				ArrayList<Integer> current = set_clusters.get((int)comp[1]).get_component();
				//math.ceil for landmarks
				//this.ouput_print.println("prob "+comp[0]*1.0/current.size()+" "+comp[1]+" "+max);
				double sim = comp[0]*1.0/current.size();
				//System.out.println(comp[0]);
				if(comp[0]<=theta || num > tau + ((Math.log(N)/Math.log(2))))
					break;


				//System.out.println("fractions is "+num_green*1.0/queries+" "+queries);
				//System.out.println("fractions i s "+num_green*1.0/queries+" "+Math.log(2)/Math.log(graph.num_nodes())+" "+set_clusters.size()+" "+queries+" "+true_pos+" "+false_pos);

				QueriedClusters.add((int)comp[1]);
				Collections.shuffle(current, new Random(rand.nextInt()));

				double prob_curr = 1.0;
				double prob_wanted = 0.9999;//Math.pow(r_to_g,num_wanted);

				double pr =1.0, pg=1.0;
				boolean addedhere = false;
				boolean first = true;
				ArrayList<Integer> currquery = new ArrayList<Integer>();
				for(int q : current){

					pair t = new pair(q,max);
					currquery.add(q);
					//System.out.println("curr"+edge_prob.get(t)+" "+avg_similarity);
					//System.out.println(q+" "+max);
					pair t11=new pair(994,1812);
					//System.out.println(t11.x+" "+t11.y+" "+edge_prob.get(t11));
					
					
					
					boolean[] output =  query_edge_prob( oracle, goldMap, q, max,0,0);
					//System.out.println(q+" "+max+" "+output[1]+" "+output[0]);
					if(!queried_edge_map.containsKey(t)){
						queries++;
						double precision = true_pos*1.0/g_edges;
						double recall = true_pos*1.0/(true_pos+false_pos);
						if(true_pos!=0)	
							System.out.println(queries+" "+true_pos+" "+false_pos+" "+precision +" "+recall+" "+2*precision*recall/(precision+recall));

						pair t1 = new pair(max,q);
						queried_edge_map.put(t1, output[1]);
						queried_edge_map.put(t, output[1]);

						//actualqueried_edge_map.put(t1, output[1]);
						//actualqueried_edge_map.put(t, output[1]);
					}


					if(output[1]){
						prob_curr *= r_to_g;
						pg *= (1 - g_to_r);
						pr *= r_to_g;		
					}
					else {
						pg *=g_to_r;
						pr *= 1 - r_to_g;
					}
					
					if(sim>=0.5 && first && output[1]){
						prob_wanted = 0.999;
						break;	
					}else if(sim < 0.5 && first && !output[1]){
						break;
					}else{

						prob_wanted = 1.0/Math.pow(Math.exp(1)*(current.size()+1), 1);
						//else
						//	prob_wanted = 0.999;
					}

					first = false;

					if(pr*1.0/(pr+pg) > this.confidence){
						break;
					}
					if(prob_curr<=prob_wanted){
						break;
					}
				}
				if(prob_curr<=prob_wanted){
					if(comp[0]<=theta)
						System.out.println("added even after low benefit");
					for(int a:currquery) {
						pair t1 = new pair(a,max);
						pair t2 = new pair(a,max);
						queried_edge_map.put(t1, true);
						queried_edge_map.put(t2, true);
					}
					for(int i:current){
						if(goldMap.get(i).equals(goldMap.get(max)))
							true_pos++;
						else{
							false_pos++;
							//pair t = new pair(i,max); 
							//System.out.println("added a wrong node"+edge_prob.get(t)+" "+current.size()+" "+max+" "+i+" "+num_g+" "+red+" "+prob_curr+" "+prob_wanted);
						}
					}

					if(!current.contains(max))
						current.add(max);
					component tmpcomp = new component(current);
					set_clusters.set((int)comp[1], tmpcomp);					
					added = true;
					actualCluster = (int)comp[1];
					nodeClusterMap.put(max,actualCluster);
					for(int a:QueriedClusters){
						if (a==actualCluster)
							continue;
						else{
							pair p1 = new pair(a,actualCluster);
							pair p2 = new pair(actualCluster,a);
							
							ClustQueryMap.put(p1,false);
							ClustQueryMap.put(p2,false);
						}
					}
					break;
				}else if(pr*1.0/(pr+pg) > this.confidence) {
					for(int a:currquery) {
						pair t1 = new pair(a,max);
						pair t2 = new pair(a,max);
						queried_edge_map.put(t1, false);
						queried_edge_map.put(t2, false);
					}
				}




				num++;
			}
			//if(less && added )
			//	System.out.println("did u see this"+max);
			if(added ==false){
			//	if(less)
			//		System.out.println("See added here only"+max);
				actualCluster = set_clusters.size();
                                nodeClusterMap.put(max,actualCluster);
                                for(int a:QueriedClusters){
                                	if (a==actualCluster)
                                        	continue;
                                                else{
                                                        pair p1 = new pair(a,actualCluster);
                                                        pair p2 = new pair(actualCluster,a);

                                                        ClustQueryMap.put(p1,false);
                                                        ClustQueryMap.put(p2,false);
                                                }
                                }
				ArrayList<Integer> curr_comp =new ArrayList<Integer> ();
				curr_comp.add(max);
				component tmpcomp = new component(curr_comp);
				set_clusters.add(tmpcomp);				
			}
			NewNodes.add(max);
			processed_nodes.add(max);
			//System.out.println(NewNodes.size()+" "+processed_nodes.size());

			if(processed_nodes.size()==N)
				break;
			if(processed_nodes.size()%100==0){
				System.out.println(true_pos+" here"+queries+" "+processed_nodes.size()+" "+neighSizelst.size()+" "+set_clusters.size());
				
				
				//if(processed_nodes.size()>400 ) 
				{
					//ModelRuleslst = TrainClassifier(neighSizelst,processed_nodes.size(),100);
				}
				double precision = true_pos*1.0/(true_pos+false_pos);
				double recall = true_pos*1.0/g_edges;
				recallprint.println(queries+" "+true_pos+" "+recall+" "+precision+" "+2*precision*recall/(precision+recall)+" "+actualqueried_edge_map.keySet().size());
				UpdateInferredEdges();
				NewNodes.clear();
				boolean updated=false;
			//	if(processed_nodes.size()==100)
				{//00==0) {
					if(pipelinetype.equals("pM3loop") ||pipelinetype.equals("ploop") )
						updated = UpdateWeight();
					System.out.println("updatig");
				}

			//	if(processed_nodes.size()==100)
				if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")|| pipelinetype.equals("nM3loop"))
					if (updated){
						GetExpected(false);
						//BlockingFinalWithoutClassifierOurPipeline.java
						if(useClassifier)
	 					ooutput =  TrainClassifierPython( newSimilarityList,0,10000000,true);//int num, boolean newsamples) throws Exception{
					}
				neighSizelst.clear();
				for(int id:recordList.keySet()){
					if(!expectedSize.containsKey(id))
						continue;
					double[] tmp = {expectedSize.get(id),id};
					neighSizelst.add(tmp);
				}
				Collections.sort(neighSizelst, new Comparator<double[]>() {
					public int compare(double[] o1, double[] o2) {
						double s1 = o1[0]; double s2 = o2[0];
						if (s1 != s2)
							return (s1 > s2 ? -1 : 1);
						else
							return 0;
					}
				});


			}

			///////
			
		}
		System.out.println(true_pos+" here"+queries+" "+processed_nodes.size()+" "+neighSizelst.size());
		this.recallprint.println(processed_nodes.size()+" "+this.N);

	}
	public void printBlockStats(){
		for (String blockName : blockIndex.keySet()){
			System.out.println(blockName+" "+blockList.get(blockIndex.get(blockName)).size());
		}
		for (double[] a :blockSize){
			System.out.println(Index2Str.get((int)a[0])+" "+a[1]+" "+a[0]);
		}
	}

	public void readWeight() throws FileNotFoundException {
		Scanner scanner = new Scanner(new File("Weighted500.txt"));
		String line;
		while(scanner.hasNextLine()){
			line= scanner.nextLine();
			line = line.toLowerCase( );
			System.out.println(line);
			String[] attributeList = line.split(" ",0);
			if(Double.valueOf(attributeList[3])<0.2)
				blockWeight.put((attributeList[1]),Double.valueOf(attributeList[7]));//weightlast*Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())/(Math.log(recordList.keySet().size()*1.0/blockList.get(blockIndex.get(slast)).size())*1.0/Math.log(recordList.keySet().size())));
			else
				blockWeight.put((attributeList[1]),(1-0.004326077071154559)*Double.valueOf(attributeList[6]));//weightlast*Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())/(Math.log(recordList.keySet().size()*1.0/blockList.get(blockIndex.get(slast)).size())*1.0/Math.log(recordList.keySet().size())));


		}
	}
	public static void main(String[] args) throws Exception{

		BlossDirtyWithoutClassifierCora FF = new BlossDirtyWithoutClassifierCora();
		System.out.println("Starting Memory KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
		
		
		String recordFile     = FF.folder+"/records.txt";
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
			int recordId = Integer.parseInt(attributeList[0].trim().replace(".jpg", ""));
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
			FF.recordList.put(recordId, line);
		}
		
		System.out.println("Number of records = "+FF.recordList.size());
		
		
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

		tmpblockList.clear();
		tmpblockList=null;
		System.gc();

		

		System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);


		String goldFile     = FF.folder+"/gold.txt";

		Scanner goldScanner = new Scanner(new File(goldFile));
		while(goldScanner.hasNextLine()){
			line= goldScanner.nextLine();
			String[] goldtokens = line.split(" ",0);
			int u = Integer.parseInt(goldtokens[0].replace(".jpg", ""));
			int id = Integer.parseInt(goldtokens[1]);
			FF.goldMap.put(u, id);
		}
		
		int id = FF.recordList.keySet().size();
		for(int i:FF.recordList.keySet()){
			FF.N++;
			if(FF.goldMap.containsKey(i))
				continue;
			else{
				FF.goldMap.put(i, id);
				id++;
			}
		}
		
		System.out.println("Memory after goldMap:" + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
		System.out.println("Going inside process");
		FF.processBlocks();

	}

}

