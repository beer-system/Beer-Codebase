/**
 * 
 */


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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;


public class feedbackframework {
	
	String folder = "dblpsm";

	HashMap<Integer,HashMap<String,Integer>> recordList = new HashMap<Integer, HashMap<String,Integer>>();
	HashMap<Integer,Double> blockWeight = new HashMap<Integer, Double>();
	HashMap<Integer,Double> blockWeightinitial = new HashMap<Integer, Double>();
	HashMap<Integer,Double> blockWeightProb = new HashMap<Integer, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	HashMap<String,Integer> blockIndex = new HashMap<String, Integer>();
	HashMap<Integer,String> Index2Str = new HashMap<Integer, String>();
	ArrayList<ArrayList<Integer>> blockList = new ArrayList<ArrayList<Integer>>();
	ArrayList<double[]> blockSize = new ArrayList<double[]>();
	HashMap<Integer,Integer> goldMap = new HashMap<Integer,Integer>();
	
	ArrayList<HashMap<Integer,Double>> recordAdjacencyList = new ArrayList<HashMap<Integer,Double>>();
	String weightType = "idf";//Other option is actual weight
	
	
	ArrayList<component> gtClusters = new ArrayList<component>();
	HashMap<Integer,Integer> gtlocation = new HashMap<Integer,Integer>();
	
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
	public void processBlocks() throws FileNotFoundException{
		
		//Initialize the adjacency list for records
		for(int i=0;i<recordList.size();i++){
			HashMap<Integer,Double> tmp = new HashMap<Integer,Double>();
			recordAdjacencyList.add(tmp);
		}
		
		
		PrintStream ouput_print = new PrintStream (folder+"/blockDist");
		
		
		//Calculate independent set blockweight
		HashMap<Integer,Boolean> pp = new HashMap<Integer,Boolean>();
		for(int i1=0;i1<recordList.keySet().size();i1++){
			if(pp.containsKey(i1))
				continue;
			ArrayList<Integer> curr = new ArrayList<Integer>();
			curr.add(i1);
			pp.put(i1, true);
			for(int i2=0;i2<recordList.size();i2++){
				if(pp.containsKey(i2))
					continue;
				else if (goldMap.get(i1).equals(goldMap.get(i2))){
					curr.add(i2);
					pp.put(i2, true);
				}
				
			}
			component t = new component(curr);
			for(int a:curr)
				gtlocation.put(a, gtClusters.size());
			gtClusters.add(t);
		}
		
		HashMap<pair,Boolean> processed = new HashMap<pair,Boolean>();
		int numIgnore = 0;
		for(int i=0;i<blockSize.size();i++){
			ArrayList<Integer> block = blockList.get((int)blockSize.get(i)[0]);
			int blockId = (int) blockSize.get(i)[0];
			int numgr=0,numrd=0;
			for(int j=0;j<block.size();j++){
				for(int k=j+1;k<block.size();k++){
					pair t1 = new pair(block.get(j),block.get(k));
					pair t2 = new pair(block.get(j),block.get(k));
					processed.put(t1, true);
					processed.put(t2, true);
					//if(blockId==5679)
						//System.out.println(block.get(j)+" "+goldMap.get(block.get(j))+" "+goldMap.get(block.get(k))+" "+(goldMap.get(block.get(j)).equals(goldMap.get(block.get(k)))));
					
					//Get the jaccard between the two and update adjacency list
					if(goldMap.get(block.get(j)).equals(goldMap.get(block.get(k))))
						numgr++;
					else
						numrd++;
				}
			}
			
			
			HashMap<Integer,Integer> clusterelem = new HashMap<Integer,Integer>();
			for(int a:block){
				if(gtlocation.containsKey(a)){
					int v = gtlocation.get(a);
					if(clusterelem.containsKey(v)){
						int val = clusterelem.get(v)+1;
						clusterelem.put(v, val);
					}else
						clusterelem.put(v, 1);
				}
			}
			int gr = 0, rd=0;
			for(int a:clusterelem.keySet()){
				gr+=clusterelem.get(a)-1;
			}
			rd+=((clusterelem.keySet().size()*(clusterelem.keySet().size()-1))/2);
			
			
			
			if (block.size()>1){
				//if(blockId==5679)
					//System.out.println(Index2Str.get(blockId)+" "+numgr+" "+numrd);
				ouput_print.println(block.size()+" "+Math.log(recordList.keySet().size()*1.0/block.size())*1.0/Math.log(recordList.keySet().size()*1.0/2)+" "+numgr*1.0/(numgr+numrd));
				if(weightType =="idf")
					blockWeight.put(blockId, Math.log(recordList.keySet().size()*1.0/block.size())*1.0/Math.log(recordList.keySet().size()*1.0/2));
				else if(weightType =="normal")
					blockWeight.put(blockId, 1.0);
				else if (weightType=="indep")
					blockWeight.put(blockId, gr*1.0/(gr+rd));
				else
					blockWeight.put(blockId, numgr*1.0/(numgr+numrd));
			}else{
				blockWeight.put(blockId, 1.0);
			}
			
			//if(numgr+numrd > 100)
				//System.out.println(i+" "+numgr*1.0/(numgr+numrd));
		}
		ouput_print.close();
		//System.out.println(processed.keySet().size());
		
		
		
		//Iterate over the pair of records to get the similarities and probabilities
		ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
		PrintStream edgesim = new PrintStream("edgesim.txt");
		for(int id1 : recordList.keySet()){
			for(int id2: recordList.keySet()){
				if(id1<id2){
					if(id1==46 && id2==388){
						System.out.println(recordList.get(id1)+" "+recordList.get(id2));
						System.out.println(blockWeight.get(blockIndex.get("st"))+" "+blockWeight.get(blockIndex.get("7200new"))+" "+blockWeight.get(blockIndex.get("212")));
						System.out.println(blockWeight.get(blockIndex.get("582"))+" "+blockWeight.get(blockIndex.get("american21"))+" "+blockWeight.get(blockIndex.get("club21")));
					}
					double[] p = {Weightedjaccard(recordList.get(id1),recordList.get(id2)), id1, id2};
					SimilarityList.add(p);
					pair t1  = new pair(id1,id2);
					pair t2  = new pair(id2,id1);
					edge_prob.put(t1, p[0]);
					edge_prob.put(t2, p[0]);
					if(p[1]==12 && p[2]==13)
						System.out.println(p[0]+" "+p[1]+" "+p[2]);
					edgesim.println((int)p[1]+" "+(int)p[2]+" "+p[0]);
					//get the similarity and isner in the list..
					//if(id1==581 && id2>581 && id2 < 590)
						//System.out.println(recordList.get(id1)+" sssssss "+recordList.get(id2));
				}
			}
		}
		
		/*HashMap<pair,Boolean> knownEdges = new HashMap<pair, Boolean>();
		for(int i=0;i<blockSize.size();i++){//String s: blockIndex.keySet()){
			
			int blockId = (int) blockSize.get(i)[0];
			String s = Index2Str.get(blockId);
			
			ArrayList<Integer> lst = blockList.get(blockIndex.get(s));
			int numgr=0;
			int numrd=0;
			int probgr=0,probrd=0;
			//DO it here
			
			for(int i1=0;i1<lst.size();i1++){
				int u = lst.get(i1);
				for(int i2=i1+1;i2<lst.size();i2++){
					int v = lst.get(i2);
					pair t = new pair(u,v);
					pair t1 = new pair(v,u);
					if(edge_prob.get(t)>=0.7){
						knownEdges.put(t, true);
						knownEdges.put(t1, true);
						numgr++;	
					}
					else if (edge_prob.get(t)<=0.3){
						numrd++;
						knownEdges.put(t1, false);
						knownEdges.put(t, false);
					}
					if(goldMap.get(u).equals(goldMap.get(v)))
						probgr++;
					else
						probrd++;
				}
			}
			int loc = blockIndex.get(s);
			if(numgr+numrd>0)
			blockWeightinitial.put(blockIndex.get(s), numgr*1.0/(numgr+numrd));
			else
			blockWeightinitial.put(blockIndex.get(s), -1.0);
			
			if(probgr+probrd>0)
				blockWeightProb.put(blockIndex.get(s), probgr*1.0/(probgr+probrd));
				else
				blockWeightProb.put(blockIndex.get(s), -1.0);
			
		}
		
		
		HashMap<Integer,Integer> Processed = new HashMap<Integer,Integer>();
		ArrayList<component> Clusters = new ArrayList<component>();
		for(int n1=0;n1<recordAdjacencyList.size();n1++){
			int loc1 = -1;
			ArrayList<Integer> comp;
			if(Processed.containsKey(n1)){
				loc1 = Processed.get(n1);
				comp = Clusters.get(loc1).get_component();
			}else{
				 comp = new ArrayList<Integer>();
				 comp.add(n1);
				 component t = new component(comp);
				 Clusters.add(t);
				 loc1 = Clusters.size()-1;
			}
			//System.out.println(n1);
			for(int n2 = n1+1;n2<recordAdjacencyList.size();n2++){
				pair t = new pair(n1,n2);
				pair t1 = new pair(n1,n2);
				int loc2 = -1;
				if(Processed.containsKey(n2)){
					loc2 = Processed.get(n2);
				}
				if(loc1==loc2)
					continue;
				if(knownEdges.containsKey(t)){
					if(knownEdges.get(t)){
						if(loc2==-1){
							comp.add(n2);
							Processed.put(n2, loc1);
						}else{
							ArrayList<Integer> temp = Clusters.get(loc2).get_component();
							comp.addAll(temp);
							temp.clear();
							for(int a:comp)
								Processed.put(a, loc1);
						}
					}
				}
			}
			component cc = new component(comp);
			Clusters.set(loc1,cc);
			
		}
		System.out.println("done it is here"+knownEdges.keySet().size()+" "+Clusters.size());
		for(int i1=0; i1<Clusters.size();i1++){
			//System.out.println(i1);
			ArrayList<Integer> comp1 = Clusters.get(i1).get_component();
			for(int a:comp1){
				for(int b:comp1){
					if(a!=b){
						pair t = new pair(a,b);
						pair t1 = new pair(a,b);
						knownEdges.put(t1, true);
						knownEdges.put(t, true);
					}
				}
			}
			for(int i2=i1+1;i2<Clusters.size();i2++){
				ArrayList<Integer> comp2 = Clusters.get(i2).get_component();
				boolean found = false;
				for(int a:comp1){
					for(int b:comp2){
						pair t = new pair(a,b);
						if(knownEdges.containsKey(t)){
							found = true;
							break;
						}
					}
					if(found)
						break;
				}
				if(found){
					for(int a:comp1){
						for(int b:comp2){
							pair t = new pair(a,b);
							pair t1 = new pair(b,a);
							knownEdges.put(t1, false);
							knownEdges.put(t, false);
						}
					}
				}
			}
			
		}
		System.out.println("done it is here"+knownEdges.keySet().size());
		*/	
			
		
		
		
		
		
		PrintStream blockAnalysis = new PrintStream("blockweights.txt");
		for(int i=0;i<blockSize.size();i++){//String s: blockIndex.keySet()){
			
			int blockId = (int) blockSize.get(i)[0];
			String s = Index2Str.get(blockId);
			
			ArrayList<Integer> lst = blockList.get(blockIndex.get(s));
			double numgr=0;
			double numrd=0;
			int probgr=0,probrd=0;
			//DO it here
			
			for(int i1=0;i1<lst.size();i1++){
				int u = lst.get(i1);
				for(int i2=i1+1;i2<lst.size();i2++){
					int v = lst.get(i2);
					pair t = new pair(u,v);
					pair t1 = new pair(v,u);
					//if(knownEdges.containsKey(t))
					{
						numgr+=edge_prob.get(t);
						numrd+=(1-edge_prob.get(t));
						/*
						if(knownEdges.get(t))
							numgr++;
						else
							numrd++;
							*/
					}
					if(goldMap.get(u).equals(goldMap.get(v)))
						probgr++;
					else
						probrd++;
				}
			}
			int loc = blockIndex.get(s);
			if(numgr+numrd>0)
			blockWeightinitial.put(blockIndex.get(s), numgr*1.0/(numgr+numrd));
			else
			blockWeightinitial.put(blockIndex.get(s), -1.0);
			
			if(probgr+probrd>0)
				blockWeightProb.put(blockIndex.get(s), probgr*1.0/(probgr+probrd));
				else
				blockWeightProb.put(blockIndex.get(s), -1.0);
			blockAnalysis.println(lst.size()+" "+blockIndex.get(s)+" "+blockWeight.get(loc)+" "+blockWeightinitial.get(loc)+" "+blockWeightProb.get(loc));
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
	   	 
	   	 
	   	 
	   	 
	   	int bucketSize =1000;//recordList.keySet().size()/2;
	   	int i=0;
	   	ArrayList<double[]> bucket =new ArrayList<double[]>();
	   	PrintStream probprint = new PrintStream (folder+"/probValues");
	   	PrintStream simprint = new PrintStream (folder+"/simValues");
	   	
	   	double max = 1.0,min = 0.99;
		for(;i<SimilarityList.size();i++){
			double[] curr = SimilarityList.get(i);
			if(curr[0]<min || i==SimilarityList.size()-1){
				if(bucket.size()>100 || i==SimilarityList.size()-1){
					int gr=0;
	   				int rd=0;
	   				for (double[] p:bucket){
	   					
	   					int u = (int) p[1];
	   					int v = (int) p[2];
	   					if(goldMap.get(u).equals(goldMap.get(v)))
	   						gr++;
	   					else
	   						rd++;
	   					//if(min == 0.99)
	   						//System.out.println(u+" "+v+" "+gr+" "+rd+" "+p[0]);
	   				}
	   				for(double[] p:bucket)
	   					probprint.println((int)p[1]+" "+(int)p[2]+" "+gr*1.0/(gr+rd));
	   					//probprint.println(bucket.get(0)[0]+" "+bucket.get(bucket.size()-1)[0]+" "+gr*1.0/(gr+rd)+" "+bucket.size());
	   				bucket.clear();
	   				max=min;
	   				min = max-0.01;
	   				bucket.add(curr);
					continue;
				}else{
					min-=0.01;
				}
			}
			bucket.add(curr);
			
		}
	   	System.out.println(SimilarityList.size());
	   	
	   	
	   /*	int num=0;
	   	for(;i<SimilarityList.size();i++){
	   		bucket.add(SimilarityList.get(i));
	   		simprint.println((int)SimilarityList.get(i)[1]+" "+(int)SimilarityList.get(i)[2]+" "+SimilarityList.get(i)[0]);
	   		num++;
	   		if(bucket.size()>bucketSize){
	   			if(i+1==SimilarityList.size() || bucket.get(bucket.size()-1)[0] == SimilarityList.get(i+1)[0]){
	   				int gr=0;
	   				int rd=0;
	   				for (double[] p:bucket){
	   					int u = (int) p[1];
	   					int v = (int) p[2];
	   					if(goldMap.get(u).equals(goldMap.get(v)))
	   						gr++;
	   					else
	   						rd++;
	   				}
	   				for(double[] p:bucket){
	   					probprint.println(p[0]+" "+gr*1.0/(gr+rd));
	   				}
	   					//	   				probprint.println(bucket.get(0)[0]+" "+bucket.get(bucket.size()-1)[0]+" "+gr*1.0/(gr+rd));
	   			//	if(bucket.get(0)[0]>0.2)
	   				System.out.println("buck" +bucket.get(0)[0]+" "+bucket.get(bucket.size()-1)[0]+" "+gr*1.0/(gr+rd)); 
	   				bucket.clear();
	   			}
	   		}
	   		if(i+1==SimilarityList.size() ){
	   			int gr=0;
   				int rd=0;
   				for (double[] p:bucket){
   					int u = (int) p[1];
   					int v = (int) p[2];
   					if(goldMap.get(u).equals(goldMap.get(v)))
   						gr++;
   					else
   						rd++;
   				}
   				for(double[] p:bucket){
   					probprint.println(p[0]+" "+gr*1.0/(gr+rd));
   				}
   				
   				//probprint.println(bucket.get(0)[0]+" "+bucket.get(bucket.size()-1)[0]+" "+gr*1.0/(gr+rd));

	   		}
	   	}//This loop converts to probability values.....
	   	
	   	
	   	*/
	   	//probprint.close();
	   	//System.out.println(num);
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
    	Random rand = new Random(1991);
    	
    	/*
    	ArrayList<Integer> largest = new ArrayList<Integer>(); 
    	Scanner clusterscanner = new Scanner(new File(folder+"/largest.txt"));
    	while (clusterscanner.hasNextLine()){
    		String line =  clusterscanner.nextLine();
    		for(String s:line.split(", ")){
    		//System.out.println(s);
    			largest.add(Integer.parseInt(s));
    		}
    	}
   	 	*/
    	
   	 	feedbackframework FF = new feedbackframework();
    	
    	String recordFile     = FF.folder+"/records.txt";
    	
   	 	Scanner scanner = new Scanner(new File(recordFile));
   	 	HashMap<String,ArrayList<Integer>> tmpblockList = new HashMap<String, ArrayList<Integer>>();
   	 	int len = 1000;
   	 	int iter=0;
   	 	while(scanner.hasNextLine()){
   	 		String line= scanner.nextLine();
   	 		line = line.toLowerCase( );
   	 		//System.out.println(line);
   	 		String[] tokenList = line.split(" ",0);
   	 		//ArrayList<String> record = new ArrayList<String>();
   	 		HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
   	 		
   	 		int recordId = iter;//Integer.parseInt(tokenList[0]);
   	 		//if(!largest.contains(recordId))
   	 		for (int i=1;i<tokenList.length;i++){
   	 			if(tokenList[i].length()<=1)
   	 				continue;
   	 			if(recordMap.containsKey(tokenList[i])){
   	 				int val = recordMap.get(tokenList[i])+1;
   	 				recordMap.put(tokenList[i],val);
   	 				continue;//Avoiding double insertion of same element in the block
   	 			}
   	 			recordMap.put(tokenList[i],1);
   	 			ArrayList<Integer> blockContent;
				if(tmpblockList.containsKey(tokenList[i])){
   	 				 blockContent = tmpblockList.get(tokenList[i]);
   	 				if( tokenList[i].equals("ginger") )
   	   	 				System.out.println(recordId+" ginger "+tokenList[i]+" "+(tokenList[i].equals("ginger") )+" "+blockContent);
   	   	 			
				}
				else
					blockContent = new ArrayList<Integer>();
				
				blockContent.add(recordId);
				tmpblockList.put(tokenList[i], blockContent);
   	 		}
   	 		/*if(largest.contains(recordId)){
   	 			for(int i=0;i<10;i++){
   	 				String newblock = "dummy1"+Integer.toString(i);
   	 				ArrayList<Integer> blockContent;
   	 				if(tmpblockList.containsKey(newblock))
   	 					blockContent = tmpblockList.get(newblock);
   	 				else
   	 					blockContent = new ArrayList<Integer>();
			
   	 				blockContent.add(recordId);
   	 				tmpblockList.put(newblock, blockContent);
   	 				recordMap.put(newblock,1);
   	 			}
   	 			int a = rand.nextInt(10);
   	 			Object[] lst =  tmpblockList.keySet().toArray();
   	 			for(int i=0;i<a;i++){
   	 				int  n = rand.nextInt(lst.length);//CHose a random block
   	 				
   	 				String blockname = (String)lst[n];
   	 				if(recordMap.containsKey(blockname))
   	 					continue;
   	 				ArrayList<Integer> blockContent;
	 				if(tmpblockList.containsKey(blockname))
	 					blockContent = tmpblockList.get(blockname);
	 				else
	 					blockContent = new ArrayList<Integer>();
		
	 				blockContent.add(recordId);
	 				tmpblockList.put(blockname, blockContent);
	 				recordMap.put(blockname,1);
   	 			}
   	 		}*/
   	 		FF.recordList.put(recordId, recordMap);
   	 		if(tokenList.length < len)
   	 			len = tokenList.length;
   	 		iter++;
   	 	}
   	 	//Free blockList and blockSize and form the lists....
   	 	int loc=0;
   	 	for (String s:tmpblockList.keySet()){
   	 		FF.blockList.add(tmpblockList.get(s));
   	 		FF.blockIndex.put(s,loc);
   	 		if(s.equals("ginger"))
   	 		System.out.println(loc+" "+s);
   	 		FF.Index2Str.put(loc, s);
   	 		double[] tmp = {loc,tmpblockList.get(s).size()};
   	 		FF.blockSize.add(tmp);
   	 		loc+=1;
   	 	}
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
   	 	iter=0;
	 	Scanner goldScanner = new Scanner(new File(goldFile));
	 	while(goldScanner.hasNextLine()){
   	 		String line= goldScanner.nextLine();
   	 		//System.out.println(line);
   	 		String[] goldtokens = line.split(" ",0);
   	 		int u = iter;//Integer.parseInt(goldtokens[0]);
   	 		int id = Integer.parseInt(goldtokens[1]);
   	 		FF.goldMap.put(u, id);
   	 		iter++;
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
   	 	System.out.println(len+"here");

   	 	FF.processBlocks();
    	
    }
    	
}

