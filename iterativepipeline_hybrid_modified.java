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
import java.util.stream.Collectors;


public class iterativepipeline_hybrid_modified {
	Random rand = new Random(1991);
	String folder = "cars2";
	PrintStream recallprint;
	HashMap<pair,Boolean> oracle = new HashMap<pair,Boolean>();

	ArrayList<component> set_clusters = new ArrayList<component>();
	double theta = 0.30;
	int tau=0;
	int numIgnore=20;//Number of blocks to ignore
	int queries=0;
	double confidence = 0.95;
	double true_pos = 0,false_pos = 0;
	String pipelinetype = "iNoloop";
	double g_edges = 2588937;//3250373;//755388;//87512;//343524;//87512;//328411//65000;
	double g_to_r=0.0;
	double r_to_g=0.0;
	Map<pair, Boolean> queried_edge_map = new HashMap<pair, Boolean>();
	Map<pair, Boolean> actualqueried_edge_map = new HashMap<pair, Boolean>();
	HashMap<Integer,HashMap<String,Integer>> recordList = new HashMap<Integer, HashMap<String,Integer>>();
	HashMap<Integer,Double> blockWeight = new HashMap<Integer, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();
	
	HashMap<String,Integer> numGreen = new HashMap<String, Integer>();
	HashMap<String,Integer> numRed = new HashMap<String, Integer>();
	
	HashMap<String,Integer> blockIndex = new HashMap<String, Integer>();
	HashMap<Integer,String> Index2Str = new HashMap<Integer, String>();
	ArrayList<ArrayList<Integer>> blockList = new ArrayList<ArrayList<Integer>>();
	ArrayList<double[]> blockSize = new ArrayList<double[]>();
	HashMap<Integer,Integer> goldMap = new HashMap<Integer,Integer>();
	
	//ArrayList<HashMap<Integer,Double>> recordAdjacencyList = new ArrayList<HashMap<Integer,Double>>();
	
	HashMap<Integer, Double> expectedSize = new HashMap<Integer, Double>();
	
	
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
	public double Weightedjaccard (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		for(String a:s1.keySet()){
			if(s2.containsKey(a)){
				inter+=(blockWeight.get(blockIndex.get(a))*Math.min(s2.get(a), s1.get(a)));
				union+=(blockWeight.get(blockIndex.get(a))*Math.max(s2.get(a), s1.get(a)));
			}else
				union+=(blockWeight.get(blockIndex.get(a))*s1.get(a));
		}
		for(String a:s2.keySet()){
			if(s1.containsKey(a))
				continue;
			else
				union+=(blockWeight.get(blockIndex.get(a))*s2.get(a));	
		}
		
		return inter*1.0/union;
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
	public boolean[] query_edge_prob(HashMap<pair,Boolean>  oracle, HashMap<Integer, Integer> gt, int u, int v, double r_to_g, double g_to_r){
		boolean ret[] = {false, false};
		pair tmp = new pair(u,v);

		/*if(oracle.get(tmp))
			ret[1]=true;
		else
			ret[1]=false;
		*/
		
		//ret[1]=false;
		//Ground truth
		if (gt.get(u).equals(gt.get(v))){
			ret[0]=true;
		}
		ret[1]=ret[0];
		return ret;
	}
	
	public boolean[] query_edge_prob( HashMap<Integer, Integer> gt, int u, int v, double r_to_g, double g_to_r){
		boolean[] ret = {false,false};
		if(gt.get(u).equals(gt.get(v))){
			ret[0]=true;
			ret[1]=true;
		}
			
		return ret;
	}
	public void UpdateGreen(int n1, int n2){
		HashMap<String,Integer> s1 = recordList.get(n1);
		HashMap<String,Integer> s2 = recordList.get(n2);
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
		HashMap<String,Integer> s1 = recordList.get(n1);
		HashMap<String,Integer> s2 = recordList.get(n2);
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
	public void UpdateWeight() throws FileNotFoundException{
		
		pair t1=  new pair(25, 513);
		System.out.println("pppppp"+edge_prob.get(t1));
		
		PrintStream bw = new PrintStream(folder+"/bw.txt");
		PrintStream bwand = new PrintStream("and.txt");
		
		for(int i=0;i<blockSize.size();i++) {
			double maxprob = 0.0;
			int queried=0;
			int blockId = (int) blockSize.get(i)[0];
			//System.out.println(blockSize.get(i)[1]+" "+blockId);
			String s = Index2Str.get(blockId);
			if(i>blockSize.size() - numIgnore && !s.equals("chevrolet") && !s.equals("bmw"))
				continue;
			ArrayList<Integer> lst  = blockList.get(blockIndex.get(s));
			s=s.replaceAll("/", "");
			PrintStream bw1 = new PrintStream("blocks/"+s);
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
						
						if(edge_prob.get(t)>maxprob)
							maxprob = edge_prob.get(t);
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
					
					bw.println(s+" "+lst.size()+" "+Math.log(recordList.keySet().size()*1.0/lst.size())*1.0/Math.log(recordList.keySet().size())+" "+gr*1.0/(gr+rd)+" "+(ng*1.0/(ng+nr))+" "+maxsize*1.0/lst.size()+" "+1.0/clustsize.keySet().size()+" "+(gr*1.0/(gr+rd))*(1.0/clustsize.keySet().size())*(maxsize*1.0/lst.size()));//+" "+maxsize+" "+clustsize.keySet().size()+" "+maxsize*1.0/lst.size());//(gr*1.0/(gr+rd)));
					//bw.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd)+" "+ng*1.0/(ng+nr)+" "+(gr*1.0/(gr+rd))*1.0/(ng*1.0/(ng+nr)));
					//blockWeight.put(blockIndex.get(s), (ng*1.0/(ng+nr)));
					/*if(lst.size()==1) {
						ng = 1.0;
						nr=1.0;
					}*/
					double weighted=0.0, normal=0.0;

					//if(gr*1.0/(gr+rd) >= 0.99 && lst.size()>1) 
					{
						for(int i11=0;i11<lst.size();i11++) {
							int u = lst.get(i11);
							for(int j=i11+1;j<lst.size();j++) {
								
								int v = lst.get(j);
								weighted += Weightedjaccard(recordList.get(u),recordList.get(v));
								normal += jaccard(recordList.get(u),recordList.get(v));
							}
						}
						weighted = weighted*2.0/(lst.size()*(lst.size()-1));
						normal = normal*2.0/(lst.size()*(lst.size()-1));
						System.out.println("block is "+s+" "+lst.size()+" "+weighted+" "+normal);
					}
					
					//bw.println(s+" "+lst.size()+" "+gr*1.0/(gr+rd)+" "+weighted+" "+normal);
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
						blockWeight.put(blockIndex.get(s), Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if(fulldummy == 0)
						blockWeight.put(blockIndex.get(s), Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy== 2)
						blockWeight.put(blockIndex.get(s), Math.pow(gr*1.0/(gr+rd),1)*((maxsize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy==3) {
						if(queried>10)
							blockWeight.put(blockIndex.get(s), Math.pow((gr*1.0/(gr+rd)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
						else
							blockWeight.put(blockIndex.get(s), Math.pow((ng*1.0/(ng+nr)),1)*((dummysize)*1.0/lst.size())*(1.0/numclust));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					}else if (fulldummy ==4)
						blockWeight.put(blockIndex.get(s), Math.pow(ng*1.0/(ng+nr),1)*((maxsize)*1.0/lst.size())*(1.0/clustsize.keySet().size()));//(maxsize*1.0/lst.size())//clustsize.keySet().size()
					else if (fulldummy==5)
						blockWeight.put(blockIndex.get(s), Math.pow(gr*1.0/(gr+rd),1));
					else if (fulldummy==6) {
						//if(lst.size()>1)
							blockWeight.put(blockIndex.get(s), (gr*1.0/(gr+rd))*Math.exp(entropy) );//(maxsize*1.0/lst.size())//clustsize.keySet().size()		
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
	public void GetExpected() throws FileNotFoundException{
		SimilarityList.clear();
		edge_prob.clear();
		
		expectedSize.clear();
		PrintStream pt = new PrintStream("blockweight.txt");
		HashMap<pair,Boolean> processed  =new HashMap<pair,Boolean>();
		for(int i=0;i<blockSize.size();i++) {
			int blockId = (int) blockSize.get(i)[0];
			System.out.println(i+" "+blockSize.get(i)[1]+" "+blockId);
			String s = Index2Str.get(blockId);
			if(i>blockSize.size() - numIgnore && !s.equals("chevrolet") && !s.equals("bmw"))
				continue;
			
			ArrayList<Integer> lst = blockList.get(blockIndex.get(s));
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
						double[] p = {Weightedjaccard(recordList.get(u),recordList.get(v)), u, v};
						SimilarityList.add(p);
						if(u==25 && (v==513))
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
		for(int i=0 ;i<SimilarityList.size();i++){
			double[] curr = SimilarityList.get(i) ;
			//if (i<100)
			//System.out.println(curr[0]+" "+curr[1]+" "+curr[2]+" "+goldMap.get((int)curr[1])+" "+goldMap.get((int)curr[2]));
			//curr[0] = curr[0]*1.0/(maxsim);
			if(curr[0] > 1.0)
				curr[0]=1.0;
			//Amplify the similarity because all similarities are in 0-0.4
			if(curr[0]<min || i==SimilarityList.size()-1){
				if(i==SimilarityList.size()-1)
					bucket.add(curr);
				if(bucket.size()>100 || i==SimilarityList.size()-1 ){
					if(numEdges<100) {
						System.out.println(min+"block values "+gr+" "+rd+" "+prevgr+" "+prevrd);
						//gr=prevgr;
						//rd=prevrd;
					}
						
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
	   					if(u==25 && v==513)
	   						System.out.println(p[0]+"fjsdkfhsdj "+pr+" "+gr+" "+rd+" "+bucket.size());
	   					edge_prob.put(t1, pr);
	   					edge_prob.put(t2, pr);
	   					//if(pr>0.5) 
	   					{
	   						
	   					
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
	   					}
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
			
			
			
			if(queried_edge_map.containsKey(t) && ! pipelinetype.equals("ploop")){
	   			if(queried_edge_map.get(t))
	   				gr++;
	   			else
	   				rd++;
	   			numEdges++;
	   		}else{
	   			//System.out.println((int)SimilarityList.get(i)[1]+" "+(int)SimilarityList.get(i)[2]);
	   			double p = SimilarityList.get(i)[0];
	   			if(curr[1]==25 && curr[2]==513)
	   			System.out.println(p+"DDDDDD"+gr+ " "+rd+" "+i);
	   			gr+=p;
				rd+=(1.0-p);
				
	   			
	   		}
	   			
			
		}
	   	
		
		/*for(int id1 : recordList.keySet()){
			double val1;
			if(expectedSize.containsKey(id1))
				val1= expectedSize.get(id1);
			else val1 = 0;
			
			for(int id2: recordList.keySet()){
				if(id1<id2){
					
					double val2;
					if(expectedSize.containsKey(id2))
						val2= expectedSize.get(id2);
					else val2 = 0;
					pair t = new pair(id1,id2);
					//System.out.println(id1+" "+id2+" ");
					if(edge_prob.containsKey(t)) {
						val2+=edge_prob.get(t);
						val1+=edge_prob.get(t);
					}
					expectedSize.put(id2, val2);
				}
			}
			expectedSize.put(id1, val1);
		}*/
		
		
	SimilarityList.clear();	
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
	
	public void processBlocks() throws FileNotFoundException{
		int N=0;
		//oracle = print_edges(goldMap);
		recallprint= new PrintStream(folder+"/recall.txt");
		//Initialize the adjacency list for records
		for(int i=0;i<recordList.keySet().size();i++){
			N++;
		}
		System.out.println(blockSize.size()+"DSDDDDDDDD"+recordList.keySet().size());
		for(int i=0;i<blockSize.size();i++){
			ArrayList<Integer> block = blockList.get((int)blockSize.get(i)[0]);
			int blockId = (int) blockSize.get(i)[0];
			if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop") || pipelinetype.equals("iM3loop")||pipelinetype.equals("iNoloop")) {
				if (block.size()>1){
					blockWeight.put(blockId, Math.log(recordList.keySet().size()*1.0/block.size())*1.0/Math.log(recordList.keySet().size()));
				}else{
					blockWeight.put(blockId, 1.0);
				}
			}else {
				blockWeight.put(blockId, 1.0);
			}	
		}
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
			if(pipelinetype.equals("pM3loop") || pipelinetype.equals("ploop"))
				UpdateWeight();	
			System.out.println("done updateweight");	
		}
		if(pipelinetype.equals("pM3loop")|| pipelinetype.equals("ploop"))
			GetExpected();
	
		
		/*for(int i=0;i<blockSize.size();i++) {
			int blockId = (int) blockSize.get(i)[0];
			
			String s = Index2Str.get(blockId);
			System.out.println(i+" "+blockSize.get(i)[1]+" "+blockId+" "+s);
		}
		if (i1>0 )
			return;
		 */
		
			System.out.println("done expected");	
	   	
		ArrayList<double[]> neighSizelst =new ArrayList<double[]>(); 
		for(int id:expectedSize.keySet()){
			double p;
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
		
		
		
		int iter=0;
		ArrayList<Integer>comlst=new ArrayList<Integer>();
		for(double[] a:neighSizelst) {
			if(goldMap.get((int)a[1]).equals(13)) {
				System.out.println(a[0]+" "+a[1]);
				comlst.add((int) a[1]);
			}
			iter++;
		}
		System.out.println(comlst.size()+" "+expectedSize.keySet().size());
		/*PrintStream testprint= new PrintStream(folder+"/test.txt");
		double[] prval= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		for(int i=0;i<comlst.size();i++) {
			for(int j=i+1;j<comlst.size();j++) {
				pair p=new pair(comlst.get(i),comlst.get(j));
				if(edge_prob.containsKey(p)) {
					prval[(int)(edge_prob.get(p)*10)] = prval[(int)(edge_prob.get(p)*10)]+1;
					testprint.println(i+" "+j+" "+comlst.get(i)+" "+comlst.get(j)+" "+edge_prob.get(p));
					testprint.println(recordList.get(comlst.get(i)));
					testprint.println(recordList.get(comlst.get(j)));
				}
					
			}
		}
		for (int i=0;i<10;i++) {
			System.out.println(i+" "+prval[i]+" "+neighSizelst.size());
		}
		if(iter>1)
			return;*/
		
		
		PrintStream nodeorder = new PrintStream("nodeorder.txt");
		while(true){
			int max=-1;
			double max_benefit=-1;
			int maxnode=-1;
			double maxb = -1;
			double clustsize=0;
			
			for (double[] nodelst : neighSizelst) {
				
				//clustsize = nodelst[0];
				int     u  = (int) nodelst[1] ;
				if(processed_nodes.contains(u))
					continue;
				else{
					max =  u;
					ArrayList<double[]> comp_list = get_benefit_component(max);
					if(comp_list.size()>0) {
						if(comp_list.get(0)[0]>maxb) {
							maxb = comp_list.get(0)[0];
							maxnode=u;
							clustsize = nodelst[0];
						}
					}else {
						maxnode=u;
						clustsize = nodelst[0];
						break;
					}
					//System.out.println("max is "+max+" "+nodelst[0]);
					break;
				}
			}
			//System.out.println(neighSizelst.get(400)[0]+" ssss"+neighSizelst.get(400)[1]);
			max=maxnode;
			nodeorder.println(max+" "+goldMap.get(max));
			if(set_clusters.size()==0){
				ArrayList<Integer> nodes = new ArrayList<Integer>();
				nodes.add(max);
				component curr_comp = new component(nodes);
				set_clusters.add(curr_comp);
				processed_nodes.add(max);	
				continue;
			}
			if(max==-1) {// || queries > 2000){
				this.recallprint.println("returned here"+neighSizelst.size());
				break;
			}
			//Write a function to return the benefit between the pair of clusters in processed if they have a queryleft
			
			ArrayList<double[]> comp_list = get_benefit_component(max);
			
			int num=0;
			boolean added = false;
			boolean less=false;
			for(double[] comp : comp_list){

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
					boolean[] output =  query_edge_prob( oracle, goldMap, q, max,0,0);
						//this.ouput_print.println(q+" "+max+" "+output[1]+" "+output[0]);
						if(!queried_edge_map.containsKey(t)){
							queries++;
							double precision = true_pos*1.0/g_edges;
							double recall = true_pos*1.0/(true_pos+false_pos);
							//if(true_pos!=0)	
								//this.ouput_print.println(queries+" "+true_pos+" "+false_pos+" "+precision +" "+recall+" "+2*precision*recall/(precision+recall));
							//this.recallprint.println(max+" "+goldMap.get(max)+" "+clustsize);
							pair t1 = new pair(max,q);
							queried_edge_map.put(t1, output[1]);
							queried_edge_map.put(t, output[1]);

							actualqueried_edge_map.put(t1, output[1]);
							actualqueried_edge_map.put(t, output[1]);
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
			/*if(processed_nodes.size()>0) {
				
				
				
				PrintStream bw1 = new PrintStream("blocksafterconvergence.txt");
				
				
				for(int i=0;i<blockSize.size()-numIgnore;i++) {
					double maxprob = 0.0;
					int blockId = (int) blockSize.get(i)[0];
					//System.out.println(blockSize.get(i)[1]+" "+blockId);
					String s = Index2Str.get(blockId);
					ArrayList<Integer> lst  = blockList.get(blockIndex.get(s));
					s=s.replaceAll("/", "");
					
					double ng=0, nr=0;
					if(numGreen.containsKey(s))
						ng = numGreen.get(s);
					if(numRed.containsKey(s))
						nr = numRed.get(s);
					for(int i11=0;i11<lst.size();i11++) {
						for(int j=i11+1;j<lst.size();j++) {
							if(goldMap.get(lst.get(i11)).equals(goldMap.get(lst.get(j))))
								ng++;
							else
								nr++;
						}
					}
					//if(ng+nr> blockList.get(blockIndex.get(s)).size()){
					bw1.println(blockSize.get(i)[0]+" "+  s+" "+ng+" "+nr+" "+ng*1.0/(ng+nr)+" "+blockWeight.get(blockIndex.get(s)));
					//System.out.println(ng+" "+nr+" "+blockWeight.get(blockIndex.get(s))+" "+set_clusters.size()+" "+blockList.get(blockIndex.get(s)).size());
				}
				return;
			}*/
			if(processed_nodes.size()==N)
				break;
			if(processed_nodes.size()%100==0){
				System.out.println(true_pos+" here"+queries+" "+processed_nodes.size()+" "+neighSizelst.size()+" "+set_clusters.size());
				double precision = true_pos*1.0/(true_pos+false_pos);
				double recall = true_pos*1.0/g_edges;
				 recallprint.println(processed_nodes.size()+" "+queries+" "+true_pos+" "+recall+" "+precision+" "+2*precision*recall/(precision+recall));
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
   
	public static void main(String[] args) throws Exception{
    		
    		iterativepipeline_hybrid_modified FF = new iterativepipeline_hybrid_modified();
    	
    		String recordFile     = FF.folder+"/records.txt";
    	
   	 	Scanner scanner = new Scanner(new File(recordFile));
   	 	HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
   	 	int len = 10000;
   	 	String[] tokenList ;
   	 	String line;
 		int val;
   	 	while(scanner.hasNextLine()){
   	 		 line= scanner.nextLine();
   	 		line = line.toLowerCase( );
   	 		tokenList = line.split(" ",0);
   	 		//ArrayList<String> record = new ArrayList<String>();

   	 	HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
   	 		int recordId = Integer.parseInt(tokenList[0]);
   	 		//if(!largest.contains(recordId))
   	 		if(recordId%500==0) {
   	 			line+=  " noise";
   	 			tokenList = line.split(" ",0);
   	 		}
   	 		for (int i=1;i<tokenList.length;i++){
   	 			if(tokenList[i].length()<=1)
   	 				continue;
   	 			if(recordMap.containsKey(tokenList[i])){
   	 				val = 1;//recordMap.get(tokenList[i])+1;
   	 				recordMap.put(tokenList[i],val);
   	 				continue;//Avoiding double insertion of same element in the block
   	 			}
   	 			recordMap.put(tokenList[i],1);
   	 			ArrayList<Integer> blockContent;
				if(tmpblockList.containsKey(tokenList[i])){
   	 				 blockContent = tmpblockList.get(tokenList[i]);
   	 			}
				else
					blockContent = new ArrayList<Integer>();
				
				blockContent.add(recordId);
				tmpblockList.put(tokenList[i], blockContent);
   	 		}
   	 		
   	 		FF.recordList.put(recordId, recordMap);
   	 		
   	 		if(tokenList.length < len)
   	 			len = tokenList.length;
   	 	}
   	 	System.out.println(FF.recordList.get(2));
   		//Free blockList and blockSize and form the lists....
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
   	 System.out.println(tmpblockList.keySet().size());
   	 	tmpblockList.clear();
   	 tmpblockList=null;
   	 System.gc();
   	 System.out.println("KB: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
   	 	
   	 Collections.sort(FF.blockSize, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				double s1 = o1[1]; double s2 = o2[1];
				if (s1 != s2)
					return (s1 > s2 ? 1 : -1);
				else
					return 0;
			}
		});
   	 
   	 
   	 
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
   	 	System.out.println("AAA"+FF.Index2Str.get(0));
   	 	//FF.printBlockStats();
   	 	System.out.println(len+"here"+" "+FF.recordList.size());

   	 	FF.processBlocks();
   	 	System.out.println(len+"here"+" "+FF.recordList.size());
    	
    }
    	
}

