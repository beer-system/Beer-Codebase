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

public class BlockingFinalWithClassifierOurPipeline {
	Random rand = new Random(1991);
	static Random generator = new Random(1991);
	String folder = "cora";
	PrintStream recallprint;
	PrintStream wongprint;
	Classifier cModel;
	ArrayList<HashMap<String,Double[]>> ModelRules;
	ArrayList<ArrayList<HashMap<String,Double[]>>> ModelRuleslst;
	HashMap<pair,Boolean> oracle = new HashMap<pair,Boolean>();
	HashMap<pair,Boolean> training;
	ArrayList<component> set_clusters = new ArrayList<component>();
	double theta = 0.0;
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
	HashMap<Integer,HashMap<String,ArrayList<Integer>>> BlockTreeMap = new HashMap<Integer,HashMap<String,ArrayList<Integer>>>();
	
	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	Map<Integer, ArrayList<String>> inverted_list = new HashMap<Integer, ArrayList<String>>();
	Instances TrainingSet;
	HashMap<String,Boolean> dict = new HashMap<String,Boolean>();

	HashMap<Integer,String> recordList = new HashMap<Integer, String>();
	HashMap<String,Double> blockWeight = new HashMap<String, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();

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

	ArrayList<Integer> goodcount = new ArrayList<Integer>();
	ArrayList<Integer> badcount = new ArrayList<Integer>();
	public double Weightedjaccardold (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		
		for(String a:s1.keySet()){
			System.out.println(a+"here "+blockWeight.containsKey(a));
			if(s2.containsKey(a)){
				//System.out.println(blockWeight.keySet());
				inter+=(blockWeight.get(a)*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight.get(a)*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight.get(a)*s1.get(a));
		}
		for(String a:s2.keySet()){
			if(s1.containsKey(a))
				continue;
			else
				union+=(blockWeight.get(a)*s2.get(a));	
		}

		return inter*1.0/union;
	}
	public double Weightedjaccard (int u, int v){
		double inter = 0, union = 0;
		ArrayList<String> l1 =  inverted_list.get(u);
		ArrayList<String> l2 =  inverted_list.get(v);
		
		for(String a:l1){
			//System.out.println(a+" "+u);
			if(l2.contains(a)){
				inter+=(blockWeight.get(a));//*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight.get(a));//*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight.get(a));//*s1.get(a));
		}
		for(String a:l2){
			if(l1.contains(a))
				continue;
			else
				union+=(blockWeight.get(a));//*s2.get(a));	
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
			double prob = 0;
			for(int n : nodes){
				pair ed = new pair(u,n);
				//System.out.println(u+" "+n+" "+edge_prob.get(ed));
				if(edge_prob.containsKey(ed)) 
					prob+=edge_prob.get(ed);
			}
			double[] entry = {prob,iter};
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
		
		System.out.println(useClassifier);
		if(useClassifier) {
			if (Test(tmp,gt.get(u).equals(gt.get(v)))>0.5)
				ret[1]=true;
			else
				ret[1]=false;
			System.out.println("classifier is used"+Test(tmp,gt.get(u).equals(gt.get(v))));
		}else
			ret[1]=ret[0];	
			
		if(ret[1]!=ret[0]) {
			wongprint.println(u+" "+v+" "+ret[1]+" "+ret[0]+" "+Test(tmp,gt.get(u).equals(gt.get(v)))+" "+edge_prob.get(tmp));
		}
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
		HashMap<String,Integer> s1 = GetDict(recordList.get(n1));
		HashMap<String,Integer> s2 = GetDict(recordList.get(n2));
		for(String s:s1.keySet()){
			if(s2.containsKey(s)){
				int val = 0;
				if(numGreen.containsKey(s))
					val = numGreen.get(s);
				numGreen.put(s, val+1);
			}	
		}
	}
	public void UpdateRed(int n1, int n2){
		HashMap<String,Integer> s1 = GetDict(recordList.get(n1));
		HashMap<String,Integer> s2 = GetDict(recordList.get(n2));
		for(String s:s1.keySet()){
			if(s2.containsKey(s)){
				int val = 0;
				if(numRed.containsKey(s))
					val = numRed.get(s);
				numRed.put(s, val+1);
			}	
		}
	}
	public void UpdateInferredEdges(){
		for(int c1=0;c1<set_clusters.size();c1++){
			//Update green and red counter between the two
			component c = set_clusters.get(c1);
			ArrayList<Integer> curr  = c.get_component();
			for(int i=0;i<curr.size();i++){
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
		}

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
						if(queried_edge_map.containsKey(t)){
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
						System.out.println("block is "+s+" "+lst.size()+" "+weighted+" "+normal+" "+i);
					}

					
					//blockWeight.put(blockIndex.get(s), Math.pow((ng*1.0/(ng+nr)),1)*(maxsize*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())
					int fulldummy = 0;
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
	
	public void UpdateWeight() throws FileNotFoundException{

		
		PrintStream bw = new PrintStream("bw.txt");
		PrintStream bwand = new PrintStream("and.txt");
		for (int i=0;i<BlockTreeMap.keySet().size();i++) {
			HashMap<String, ArrayList<Integer>> level = BlockTreeMap.get(i);
			int j=0;
			for(String s:level.keySet()) {
				
				ArrayList<Integer> lst  = level.get(s);
				Collections.shuffle(lst);
				lst.subList(0, Math.min(500,lst.size()));
				
				if (j>5185 && !s.equals("chevrolet")) {
					blockWeight.put(s,0.0);//weightlast*Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())/(Math.log(recordList.keySet().size()*1.0/blockList.get(blockIndex.get(slast)).size())*1.0/Math.log(recordList.keySet().size())));

					continue;
				}
				j++;

				double ng=0, nr=0;
				if(numGreen.containsKey(s))
					ng = numGreen.get(s);
				if(numRed.containsKey(s))
					nr = numRed.get(s);
				int queried = (int) (ng+nr);
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
							if(queried_edge_map.containsKey(t)){
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
							System.out.println("block is "+s+" "+lst.size()+" "+weighted+" "+normal+" "+i);
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


						//if(lst.size()>200)
						//blockWeight.put(blockIndex.get(s), (gr*1.0/(gr+rd)));
						//else
						//	blockWeight.put(blockIndex.get(s), Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size()));

					}
				}
				
			}
		}
		
		
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

		edge_prob.clear();
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
	
	public void GetExpected() throws FileNotFoundException{
		System.out.println(queried_edge_map.keySet().size()+"size is ");
		SimilarityList.clear();
		edge_prob.clear();
		expectedSize.clear();		
		
		PrintStream pt = new PrintStream("blockweight.txt");
		
		HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
		for(int i=0;i<blockSize.size()-numIgnore;i++) {
			System.out.println(i+" "+blockSize.size());
			
			int blockId = (int) blockSize.get(i)[0];

			String s = Index2Str.get(blockId);
			
			ArrayList<Integer> lst = BlockTreeMap.get(0).get(s);

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
						double[] p = {Weightedjaccard(u,v), u, v};
						SimilarityList.add(p);
						processed.put(t1, true);
						processed.put(t, true);
					}
				}
			}

			if(lst.size()>1)
				pt.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd));
		}
		pt.close();
		processed.clear();
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

		edge_prob.clear();
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

	public ArrayList<double[]> get_benefit_clusters(){
		ArrayList<double[]> clust_benefit = new ArrayList<double[]>();
		for(int i=0;i<set_clusters.size();i++) {
			ArrayList<Integer> clus1 = set_clusters.get(i).get_component();
			for(int j=i+1;j<set_clusters.size();j++) {
				ArrayList<Integer> clus2 = set_clusters.get(j).get_component();
				boolean found = false;
				double benefit=0.0;
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
	public HashMap<String,Double> Extractfeat(pair p, Boolean cl){
		
		String[] s1 = (recordList.get(p.x)).split(";",0);
		String[] s2 = recordList.get(p.y).split(";",0);
		ArrayList<Double> featList = new ArrayList<Double>();
		//System.out.println(p.x+" "+p.y);
		for (int i=1;i<8;i++) {
			if(s1[i].equals("") || s2[i].equals("")) {
				featList.add(0.0);
				continue;
			}
			//System.out.println(i+" "+s1[i]+" "+s2[i]);
			Double val = Weightedjaccardold(GetDict("1 "+s1[i]),GetDict("2 "+s2[i]));
			if(val.isNaN())
				val=0.0;
			featList.add(val);
		}
		
	 		HashMap<String,Double> featMap = new HashMap<String,Double>();
	 		featMap.put("title", (featList.get(0)));
	 		featMap.put("authors", (featList.get(1)));
	 		featMap.put("year", (featList.get(2)));
	 		featMap.put("journal", (featList.get(3)));
	 		featMap.put("volume", (featList.get(4)));
	 		featMap.put("pages", (featList.get(5)));
	 		featMap.put("misc", (featList.get(6)));
	 		if (cl)
	 			featMap.put("Class", 1.0);
	 		else
	 			featMap.put("Class", 0.0);
	 		
	 		
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
		
		for(pair p:actualqueried_edge_map.keySet()) {
			training.put(p, actualqueried_edge_map.get(p));
		}
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
					 curr.setValue((Attribute)fvWekaAttributes.elementAt(7), edge_prob.get(p));
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
	public ArrayList<ArrayList<HashMap<String,Double[]>>> TrainClassifier(ArrayList<double[]> neighSizelst, int start,int num, boolean newsamples) throws Exception {
		
		/*double numelem = neighSizelst.get(0)[0];
		for (double[] lst: neighSizelst) {
			
		}*/
		
		training = new HashMap<pair,Boolean>();
		if(newsamples) {
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
		}
		
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
			 
			 for(String s2: feat.keySet())
				 System.out.println(s2+" "+feat.get(s2));
			 for(String s2: rule.keySet())
				 System.out.println(s2+" "+rule.get(s2));
			  
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
			if (edge_prob.containsKey(p))
			feat.put("sim", edge_prob.get(p));
			else
				feat.put("sim", 0.0);
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
			if(recordList.get(rec_id).contains("michigan") )
				System.out.println("**********found"+" "+recordList.get(rec_id)+" "+rec_id);
			
			String[] rec = recordList.get(rec_id).replace(";","").replaceAll("( )+", " ").split(" ",0);
			
			String out="";
			for (int a=0;a<rec.length;a++) {
				if (rec[a].length()>1)
				out += rec[a]+" ";
				
			}
			
			rec = out.replace(";","").replaceAll("( )+", " ").split(" ",0);
			
			
			int i=0;
			//System.out.println(recordList.get(rec_id).replace(";","").replaceAll("( )+", " "));
			for(;i<rec.length;i++) {
			
				int j=i+1;
				while (j<=rec.length) {
					String token = String.join("!", Arrays.copyOfRange(rec, i, j));
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
						System.out.println(s);
						double[] tmp = {loc,level_map.get(s).size()};
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
	public void CheckLowconfidence () {
		
		while(true) {
			boolean found=false;
			for(int i=0;i<set_clusters.size();i++) {
				ArrayList<Integer> clust = set_clusters.get(i).get_component();
				for(int j=0;j<clust.size();j++) {
					double prob=0.0;
					int pos=0,neg=0;
					for(int k=0;k<clust.size();k++) {
						pair p=new pair(clust.get(j),clust.get(k));
						if (edge_prob.containsKey(p))
							prob+=edge_prob.get(p);
						if (j!=k) {
							if (goldMap.get(clust.get(j)).equals(goldMap.get(clust.get(k))))
								pos++;
							else
								neg++;	
						}
						
					}
					prob = prob*1.0/(clust.size()-1);
					
					if (prob<0.5) {
						
						int n2 = clust.get((j+1)%clust.size());
						pair p=new pair(n2,clust.get(j));
						pair p1=new pair(clust.get(j),n2);
						if (actualqueried_edge_map.containsKey(p) || actualqueried_edge_map.containsKey(p1))
							continue;
						if (goldMap.get(p.x).equals(goldMap.get(p.y)))
							actualqueried_edge_map.put(p, true);
						else {
							actualqueried_edge_map.put(p, false);
							found=true;
							
							ArrayList<Integer> clrlist = new ArrayList<Integer>();
							clrlist.add(clust.get(j));
							component clrtmp = new component(clrlist);
							set_clusters.add(clrtmp);
							
							
							clust.remove(j);
							component newtmp = new component(clust);
							set_clusters.set(i, newtmp);
							
							
							true_pos-=pos;
							false_pos-=neg;
							break;
						}
						recallprint.println("less confidence edge"+prob+" "+clust.get(j)+" "+goldMap.get(clust.get(j))+" "+goldMap.get(clust.get(0)));
						
					}
				}
			}
			if(!found)
				break;
		}
		
	}
	public boolean mergeClusters( HashMap<Integer, Integer> gt, Map<pair, Double> edge_prob, int g_edges, double maxbenefit) throws Exception{
		
		
		int num_clust_queried = 0;
		int logn = (int) ((int)Math.log(recordList.size())/Math.log(2));
		
		 while(true){
			 
			if(num_clust_queried > (logn*(logn-1))/2)
				break;
			
			List<double[]> probList = new ArrayList<double[]>();
			for(int i=0;i<set_clusters.size();i++){
				for(int j=i+1;j<set_clusters.size();j++){
					ArrayList<Integer> a = set_clusters.get(i).get_component();
					ArrayList<Integer> b = set_clusters.get(j).get_component();
					int pos=0,neg=0;
					double pg = 1.0, pr=1.0;
					double benefit = 0.0;
					int num_q=0;
					
					for(int x:a){
						for(int y:b){
							pair t  = new pair(x,y);
							pair t1  = new pair(y,x);
							if(actualqueried_edge_map.containsKey(t) ){
								num_q++;
								if(actualqueried_edge_map.get(t)){
									pg *= (1-g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
									pr *= r_to_g;//(edge_prob.get(t)*max_r_to_g);
								}else{
									pg *= (g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
									pr *= 1-r_to_g;//(edge_prob.get(t)*max_r_to_g);
								}	
							}
							else {
								if(actualqueried_edge_map.containsKey(t1) ){
									num_q++;
									if(actualqueried_edge_map.get(t1)){
										pg *= (1-g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
										pr *= r_to_g;//(edge_prob.get(t)*max_r_to_g);
									}else{
										pg *= (g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
										pr *= 1-r_to_g;//(edge_prob.get(t)*max_r_to_g);
									}	
								}
							}
							if(edge_prob.containsKey(t))
								benefit+=edge_prob.get(t);
							if(gt.get(x).equals(gt.get(y)))
								pos++;
							else neg++;
						}
					}
					if (pr*1.0/(pr+pg)<this.confidence && (pg<1.0 || pr<1.0))
						continue;
					if(a.size()==0)
						continue;
					if(b.size()==0)
						continue;
					if(benefit<= maxbenefit)
						continue;
					
					double[] entry = {benefit,i,j,pos,neg,pr,pg, benefit/(a.size()*b.size()), num_q};
					if(entry[7]<0.5)
						continue;
					
					
					probList.add(entry);
				}
			}
			System.out.println(probList.size());
			if(probList.size()==0)
				break;
			Collections.sort(probList, new Comparator<double[]>() {
				public int compare(double[] o1, double[] o2) {
					double s1 = o1[0]; double s2 = o2[0];
					if (s1 != s2)
						return (s1 > s2 ? -1 : 1);
					else
						return 0;
				}
			});

			for(double[]entry : probList){
				int i=(int)entry[1];
				int j=(int)entry[2];
				
				if(num_clust_queried > (logn*(logn-1))/2)
					break;
				

				System.out.println(entry[0]+" "+" "+entry[1]+" "+entry[2]+" "+entry[3]+" "+entry[4]+" "+entry[5]+" "+entry[6]+" "+entry[7]);
				
				ArrayList<Integer> a = set_clusters.get((int)entry[1]).get_component();
				ArrayList<Integer> b = set_clusters.get((int)entry[2]).get_component();
				//this.ouput_print_scc.println("trying for "+logn+ " "+a.size()+" "+b.size()+" "+num_clust_queried);		
				
					num_clust_queried++;
					
					double prob_curr = 1.0;
					int size = Math.min(a.size(), b.size());
					double prob_wanted = 1.0;//1.0/Math.pow(Math.exp(1)*(a.size()+b.size()), 1);
					prob_wanted = Math.pow(prob_wanted, size);
					double pr=entry[5],pg=entry[6];
					
					if(pr*1.0/(pr+pg) > this.confidence)
						continue;
					if(entry[8] == a.size() * b.size())
						continue;
					
					ArrayList<int[]> pairlist = new ArrayList<int[]>();
					for(int a1:a){
						for(int a2:b){
							int[] tmp  = {a1,a2};
							pairlist.add(tmp);
						}
					}
					Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
					Collections.shuffle(pairlist,newgen);
					
					for(int[] querypair:pairlist){
						pair t = new pair(querypair[0],querypair[1]);
						pair t1 = new pair(querypair[1], querypair[0]);

						boolean[] output = {gt.get(t.x).equals(gt.get(t.y)),gt.get(t.x).equals(gt.get(t.y))};// query_edge_prob(oracle, gt, querypair[0], querypair[1],0.0,0.0);
						if(!queried_edge_map.containsKey(t)){
							queries++;
							double precision = true_pos*1.0/g_edges;
							double recall = true_pos*1.0/(true_pos+false_pos);
							
							queried_edge_map.put(t1, output[1]);
							queried_edge_map.put(t, output[1]);
							
						}
						recallprint.println("querying"+entry[1]+" "+entry[2]+" "+entry[0]+" "+entry[5]+" "+entry[6]+" "+a.size()+" "+b.size()+" "+probList.size());
						if(!actualqueried_edge_map.containsKey(t)){
							queries++;
							double precision = true_pos*1.0/g_edges;
							double recall = true_pos*1.0/(true_pos+false_pos);
							
							actualqueried_edge_map.put(t1, output[1]);
							//actualqueried_edge_map.put(t, output[1]);
							
						}
						{
							if(output[1]){
								prob_curr  *= r_to_g;//(edge_prob.get(t)*max_r_to_g);
								pg *= (1-g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
								pr *= r_to_g;//(edge_prob.get(t)*max_r_to_g);
							}else{
								pg *= (g_to_r);//(1 - (1-edge_prob.get(t))*max_g_to_r);
								pr *= (1-r_to_g);//(edge_prob.get(t)*max_r_to_g);
							}	
							
						}
						
						
						if(prob_curr<=prob_wanted ){
							break;
						}
						if(pr*1.0/(pr+pg)>this.confidence){
							break;
						}

					}
					
					if((prob_curr<= prob_wanted   )){// && tmp.size() < current.size()) || (num_g>= Math.ceil((tmp.size()*(1-g_to_r)/2)) && tmp.size() == current.size())){
						int pos=0,neg=0;
						for(int x:a){
							for(int y:b){
								if(gt.get(x).equals(gt.get(y)))
									pos++;
								else neg++;
							}
						}


						a.addAll(b);
						component newtmp = new component(a);
						set_clusters.set(i, newtmp);
						ArrayList<Integer> clrlist = new ArrayList<Integer>();
						component clrtmp = new component(clrlist);
						set_clusters.set(j, clrtmp);
						true_pos+=pos;
						false_pos+=neg;
						if(entry[3]>100)
							System.out.println("merged"+entry[0]+" "+" "+entry[1]+" "+entry[2]+" "+entry[3]+" "+entry[4]+" "+entry[5]+" "+entry[6]);
						break;
					}
				}


		}
		double recall = true_pos*1.0/g_edges;
		double precision = true_pos*1.0/(true_pos+false_pos);
		recallprint.println(set_clusters.size()+" "+queries+" "+true_pos+" "+recall+" "+precision+" "+2*precision*recall/(precision+recall)+" "+actualqueried_edge_map.keySet().size());
		return true;
	}


	public boolean large_clusters() {
		//if there are two clusters which are large and merging has high benefit and high prob then ask the crowd
		//If crowd says it is to be merged then we need to retrain
		//Similarly make a list of nodes which have low confidence with a cluster and see if they should be removed?
		//Just add these examples in the training set and update the classifier...
		for(int i=0;i<set_clusters.size();i++) {
			ArrayList<Integer> c1 = set_clusters.get(i).get_component();
			for(int j=i+1;j<set_clusters.size();j++) {
				ArrayList<Integer> c2 = set_clusters.get(j).get_component();
				double score=0.0;
				System.out.println(c1.size()+" "+c2.size());
				int gr=0;
				for (int a:c1) {
					for(int b:c2) {
						pair p=new pair(a,b);
						if (edge_prob.containsKey(p))
							score += edge_prob.get(p);
						if(goldMap.get(a).equals(goldMap.get(b)))
							gr++;
					}
				}
				
				if(score *1.0/(c1.size()*c2.size())>0.5) {
					wongprint.println("retriaining "+score *1.0/(c1.size()*c2.size())+" "+gr+" "+gr*1.0/(c1.size()*c2.size()));
					//Try to merge here	
					
				}
			}
		}
		return false;
	}
	public void processBlocks() throws Exception{
		if (mergeBlocks()==0) {
			System.out.println("done merging");
			//return;
		}
		
		
		recallprint= new PrintStream(folder+"/recall.txt");
		wongprint=new PrintStream(folder+"/wrong.txt");
		PrintStream Nodeorder= new PrintStream("nodeorder.txt");
		
//		HashMap<String,Double> ruleBlockWeight = new HashMap<String,Double>();

		System.out.println(blockSize.size()+"DSDDDDDDDD"+recordList.keySet().size());
		int blockId=0;		
		for(int i=0;i<BlockTreeMap.keySet().size();i++) {
			HashMap<String,ArrayList<Integer>> level = BlockTreeMap.get(i);
			for(String s:level.keySet()) {
				if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
					blockWeight.put(s, Math.log(recordList.keySet().size()*1.0/level.get(s).size())*1.0/Math.log(recordList.keySet().size()));
				}else {
					blockWeight.put(s, 1.0);
				}
			}	
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
		Set<Integer> processed_nodes = new HashSet<Integer>();
		int i1=0;
		for( int i=0;i<1;i++){
			i1=1;
			System.out.println(i);
			GetExpected();
			System.out.println("done expected");	
			//if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop"))
			//	readWeight();//UpdateWeight();	
			System.out.println("done updateweight");	
		}
		if(pipelinetype.equals("pM3loop")|| pipelinetype.equals("ploop"))
			GetExpected();
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

		ModelRuleslst = TrainClassifier(neighSizelst,processed_nodes.size(),100, true);
		useClassifier=true;
		double maxbenefit=0.0;
		while(true){
			
			int max=-1;
			double max_benefit=-1;
			int maxnode=-1;
			double maxb = -1;
			
			int num1=0;
			for (double[] nodelst : neighSizelst) {
				int     u  = (int) nodelst[1] ;
				max_benefit=nodelst[0];
				if(goldMap.get(u)==8)
					continue;
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
				component curr_comp = new component(nodes);
				set_clusters.add(curr_comp);
				processed_nodes.add(max);	
				continue;
			}
			if(max==-1)// || queries > 2000)
				break;
			//Write a function to return the benefit between the pair of clusters in processed if they have a queryleft

			ArrayList<double[]> comp_list = get_benefit_component(max);
			maxbenefit=comp_list.get(0)[0];
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

				if(comp[0]<=theta || num > tau + ((Math.log(N)/Math.log(2))))
					break;


				//System.out.println("fractions is "+num_green*1.0/queries+" "+queries);
				//System.out.println("fractions i s "+num_green*1.0/queries+" "+Math.log(2)/Math.log(graph.num_nodes())+" "+set_clusters.size()+" "+queries+" "+true_pos+" "+false_pos);

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
					System.out.println(q+" "+max);
					pair t11=new pair(994,1812);
					System.out.println(t11.x+" "+t11.y+" "+edge_prob.get(t11));
					
					
					
					boolean[] output =  query_edge_prob( oracle, goldMap, q, max,0,0);
					System.out.println(q+" "+max+" "+output[1]+" "+output[0]);
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
			if(less && added )
				System.out.println("did u see this"+max);
			if(added ==false){
				if(less)
					System.out.println("See added here only"+max);
				ArrayList<Integer> curr_comp =new ArrayList<Integer> ();
				curr_comp.add(max);
				component tmpcomp = new component(curr_comp);
				set_clusters.add(tmpcomp);				
			}
			processed_nodes.add(max);
			//System.out.println(processed_nodes.size());

			if(processed_nodes.size()==N)
				break;
			if(processed_nodes.size()%100==0){
				System.out.println(true_pos+" here"+queries+" "+processed_nodes.size()+" "+neighSizelst.size()+" "+set_clusters.size());
				for(ArrayList<HashMap<String,Double[]>> Tree: ModelRuleslst) {
					 for(HashMap<String,Double[]> r : Tree) {
						 System.out.println(r.keySet()+" "+r.values());
					 }
					 System.out.println("new tree");
				}
					 
				//Q1: When should I retrain the classifier
				//Q2: When should I send the feedback for re-weighting
				
				//Retrain: When classifier is not able to decide well about a cluster
				// Classifier is just adding singleton nodes
				//Boost f-score can be called if benefit is high later
				//Classifier is turning high benefit and high avg prob edges to be false...
				//if(processed_nodes.size()>400 )
				double precision = true_pos*1.0/(true_pos+false_pos);
				double recall = true_pos*1.0/g_edges;
				
				recallprint.println(set_clusters.size()+" "+queries+" "+true_pos+" "+recall+" "+precision+" "+2*precision*recall/(precision+recall)+" "+actualqueried_edge_map.keySet().size());
				CheckLowconfidence();
				recallprint.println("low confidence check done");

				if(mergeClusters(goldMap,edge_prob,(int)g_edges,maxbenefit))
				{
					recallprint.println("learning a new one");
					ModelRuleslst = TrainClassifier(neighSizelst,processed_nodes.size(),100,false);
				}
				
				UpdateInferredEdges();
				// if(processed_nodes.size()>=1000)
				{//00==0) {
					if(pipelinetype.equals("pM3loop") ||pipelinetype.equals("ploop") )
						UpdateWeight();
					System.out.println("updatig");
				}
				
				if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")|| pipelinetype.equals("nM3loop"))
					GetExpected();
				 
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

		BlockingFinalWithClassifierOurPipeline FF = new BlockingFinalWithClassifierOurPipeline();
		System.out.println("Starting Memory KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
		
		
		String recordFile     = FF.folder+"/recordsStructured.txt";
		Scanner scanner = new Scanner(new File(recordFile));
		
		HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
		String[] attributeList ;
		String line;
		int val;
		while(scanner.hasNextLine()){
			line= scanner.nextLine();
			line = line.toLowerCase( );
			//System.out.println(line);
			attributeList = line.split(" ",0);
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

