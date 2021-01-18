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


public class Falcon2017CleanClean {
	String folder = "products";
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
	HashMap<Integer,String> recordList2 = new HashMap<Integer, String>();
	HashMap<Integer,String> recordList1 = new HashMap<Integer, String>();
	 ArrayList<String> feature_names = new ArrayList<String>();
	HashMap<String,Double> blockWeight1 = new HashMap<String, Double>();
	Map<pair, Double> edge_prob = new HashMap<pair,Double>();
	ArrayList<double[]> SimilarityList = new ArrayList<double[]>();
	//HashMap<Integer,Integer> location = new HashMap<Integer,Integer>();
	
	HashMap<String,Integer> numGreen = new HashMap<String, Integer>();
	HashMap<String,Integer> numRed = new HashMap<String, Integer>();
	
	HashMap<pair,Boolean> goldMap = new HashMap<pair,Boolean>();
	
	//ArrayList<HashMap<Integer,Double>> recordAdjacencyList = new ArrayList<HashMap<Integer,Double>>();
	
	HashMap<Integer, Double> expectedSize = new HashMap<Integer, Double>();
	
	
	Map<Integer, ArrayList<String>> inverted_list1 = new HashMap<Integer, ArrayList<String>>();
	Map<Integer, ArrayList<String>> inverted_list2 = new HashMap<Integer, ArrayList<String>>();
	
	
	
	HashMap<String,Integer> blockIndex = new HashMap<String, Integer>();
	HashMap<Integer,String> Index2Str = new HashMap<Integer, String>();
	ArrayList<ArrayList<Integer>> blockList = new ArrayList<ArrayList<Integer>>();
	ArrayList<double[]> blockSize = new ArrayList<double[]>();
	
	HashMap<String,ArrayList<Integer>> blockMap1  = new HashMap<String,ArrayList<Integer>>();
	HashMap<String,ArrayList<Integer>> blockMap2  = new HashMap<String,ArrayList<Integer>>();
	public ArrayList<double[]> samplePairs(int num){
		ArrayList<double[]> samplePairs = new ArrayList<double[]>();
		
		for(int a:recordList1.keySet()) {
			for(int b:recordList2.keySet()) {
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

		for(int a:recordList1.keySet()) {
			acopy.add(a);
		}

		for(int b:recordList2.keySet()) {
			bcopy.add(b);
		}

		
		Random newgen = new Random((int)(generator.nextDouble()*Integer.MAX_VALUE));
		
		Collections.shuffle(bcopy,newgen);
		for(int i=0;i<num*1.0/y;i++) {
			//System.out.println(i+"curr iteration");
			String curr = recordList2.get(bcopy.get(i));
			String[] attributeList = curr.split(";",0);
			ArrayList<Integer> sharedRec=new ArrayList<Integer>();
			
		//	int recordId = Integer.parseInt(attributeList[0].trim());
			for (int k=1;k<attributeList.length;k++){
				String[] tokenList = attributeList[k].split(" ",0);
				for(int j=0;j<tokenList.length;j++) {
					String blockname= tokenList[j];
					if(!blockMap1.containsKey(blockname))
						continue;
					ArrayList<Integer> blockContent= blockMap1.get(blockname);
					if (blockContent.size()>0)
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
				double[] tmp1 = {blockorder.get(j)[0],bcopy.get(i)};//, blockorder.get(j)[0]};
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
					double[] tmp1 = {acopy.get(j),bcopy.get(i)};//, acopy.get(j)};
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
	

	public double Weightedjaccardold (HashMap<String,Integer> s1, HashMap<String,Integer> s2){
		double inter = 0, union = 0;
		
		for(String a:s1.keySet()){
			//System.out.println(a+"here "+blockWeight.containsKey(a));
			if(!blockWeight1.containsKey(a)) {
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
		double v1 = 0.0;
		if(!a.equals(""))v1=Double.valueOf(a);
		double v2 = 0.0;
		if( b != null && b.length() > 0)v2=Double.valueOf(b);
		if(v1==v2)
			featMap.put("exact_match", 1.0);
		else
			featMap.put("exact_match", 0.0);
		
		//featMap.put("absolute_diff", Math.abs(v1-v2));
		//featMap.put("relative_diff", Math.abs((v1-v2)*1.0/v1));
		//featMap.put("levenshtein", Levenshtein_sim (a,b) );
		
		return featMap;
	}
	public HashMap<String,Double> GetFeatures(int a, int b,int m, Boolean cl){
		pair p = new pair(a,b);
		//System.out.println(a+" "+b);
		String[] s1 = (recordList1.get(a)).split(";",0);
		String[] s2 = recordList2.get(b).split(";",0);
		ArrayList<Double> featList = new ArrayList<Double>();
		
		HashMap<String,Double> featMap = new HashMap<String,Double>();
		String rec1 = "1 ",rec2="2 ";	
		//System.out.println(p.x+" "+p.y+" "+s1.length+" "+s2.length);
		//System.out.println(recordList1.get(p.x)+" "+recordList2.get(p.y));
		for (int i=1;i<7;i++) {
			if(s1.length<=i || s2.length<=i) {
				featList.add(0.0);
				HashMap<String,Double> CurrFeatMap;
				if (i==1 || i==3) {
					CurrFeatMap  = Get_small_string_features("","");
					//System.out.println(i+" "+TitleFeatMap);
				}else if  (i!=5)  {
					CurrFeatMap  = Get_Single_Word_features("","");
				}else {
					CurrFeatMap  = Get_Single_Word_features("","");
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
					featList.add(val);
				}
				
			
			
//			System.out.println(recordList1.get(p.x)+" "+s1);
			HashMap<String,Double> CurrFeatMap;
			if (i==1 || i==3) {
				CurrFeatMap  = Get_small_string_features(s1[i],s2[i]);
				//System.out.println(i+" "+TitleFeatMap);
			}else if(i!=5) {
				CurrFeatMap  = Get_Single_Word_features(s1[i],s2[i]);
			}else {
				CurrFeatMap  = Get_Single_Word_features(s1[i],s2[i]);
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
	 		featMap.put("volume", (featList.get(4)));
	 		featMap.put("pages", (featList.get(5)));
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
	public ArrayList<HashMap<String,Double>> genFunctions(ArrayList<double[]> samplePairs, int m) {
		//Generate m features
		ArrayList<HashMap<String,Double>> featList = new ArrayList<HashMap<String,Double>>();
		for(double[] sample :samplePairs) {
			pair p = new pair((int)sample[0],((int)sample[1]));
			boolean out = goldMap.containsKey(p);
			HashMap<String,Double> featMap = GetFeatures((int)sample[0],(int)sample[1],m,out);
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
				 if(s.equals("Class"))
					 continue;
				 if(s.equals("id1") || s.equals("id2"))
					 continue;
				 Attribute tmp = new Attribute(s);
				 features.add(tmp);
				 feature_names.add(s);
			 }
			 // Declare the class attribute along with its values
			 FastVector classLabel = new FastVector(2);
			 classLabel.addElement("red");
			 classLabel.addElement("green");
			 Attribute ClassAttribute = new Attribute("Class", classLabel);
			 
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
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(i), rowfeat.get(feature_names.get(i)));
					
				 }
				 if(rowfeat.get("Class")>0)
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
						 curr.setValue((Attribute)fvWekaAttributes.elementAt(j), rowfeat.get(feature_names.get(j)));
				 }
				 if(rowfeat.get("Class")>0)
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
			 
			 System.out.println(totalConfidence+" confidence"+V.size()+" "+confidenceLst.size()+" "+AlreadyQueried.size());

			 if (processConfidenceList(confidenceLst,window,nconverged,nhigh,ndegrade,epsilon))// int window, int nconverged,int nhigh, int ndegrade, double epsilon
				 break;
			 System.out.println(totalConfidence+" confidence"+V.size()+" "+confidenceLst.size()+" "+AlreadyQueried.size());
			 for(int i=0;i<confidenceLst.size();i++)
				 System.out.println("confidence" +i+" "+confidenceLst.get(i));

			 //Done by me to give it extra edge
			/* if(!Blocking && totalConfidence>0.9)
				 break;
			 if (Blocking &&totalConfidence==1.0)
				 break;
			*/ 
			 
			 
			 
			 if(classificationNeeded) {
				 
				 ArrayList<double[]> confusingLst = new ArrayList<double[]>();
				 for(int i=0;i<featureLst.size();i++) {
					 HashMap<String,Double>rowfeat= featureLst.get(i);
					 DenseInstance curr = new DenseInstance(featNumbers);
					 
					  for(int j=0;j<features.size();j++) {
							 curr.setValue((Attribute)fvWekaAttributes.elementAt(j), rowfeat.get(feature_names.get(j)));
						
					 }
					 if(rowfeat.get("Class")>0)
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
								 if(!queried_edge_map.containsKey(tmp)) {
									 queries++;
									 queried_edge_map.put(tmp, goldMap.containsKey(tmp));
									 //queried_edge_map.put(tmp1, goldMap.get(tmp1.x).equals(goldMap.get(tmp1.y)));
									 System.out.println("queries is1 "+queries);
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
		System.out.println("I am returning");
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
			ruleMap.put("Class", val);
			
			lst.add(ruleMap);
			return lst;
			
		}else {
			//System.out.println(t.m_SplitPoint+" "+t.m_Attribute+" "+feature_names.get((t.m_Attribute)));
			//text.append(m_Info.attribute(m_Attribute).name() + " < "
		      //      + Utils.doubleToString(m_SplitPoint, getNumDecimalPlaces()));
		     
			//System.out.println(t.m_Successors.length);
			ArrayList<HashMap<String,Double[]>>  o1 =  getRules(t.m_Successors[0]);
			
			ArrayList<HashMap<String,Double[]>>  o2 =  getRules(t.m_Successors[1]);
			
			Double[] l = {-1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o1) {
				map.put(feature_names.get(t.m_Attribute)+";"+Math.random(),l);
				lst.add(map);
			}
			
			//System.out.println(t.getLocalModel().rightSide(1,t.getTrainingData()).split(" ",0)[2]);
			Double[] r = {1.0,  t.m_SplitPoint};
			for(HashMap<String,Double[]> map : o2) {
				map.put(feature_names.get(t.m_Attribute)+";"+Math.random(),r);
				lst.add(map);
			}
			
			return lst;
		}
	
		//return ruleMap;
	} 
	public boolean CheckNo(HashMap<String,Double[]> rule) {
		if(rule.get("Class")[1]>0)
			return false;
		else
			return true;
	}
	public boolean satisfy(HashMap<String,Double> feat, HashMap<String,Double[]> rule) {
		
		for(String s:rule.keySet()) {
			 if(s.trim().equals("Class"))
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
							if(row.get("Class")>0)
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
					if(iter>=ruleSatisfier.size()) {
						exit=true;
						break;
					}
					if(X.size()==ruleSatisfier.size()) {
						exit=true;
						break;
					}
					if(X.contains(ruleSatisfier.get(iter)))
						continue;
					X.add(ruleSatisfier.get(iter));
					pair p = new pair((featureLst.get(ruleSatisfier.get(iter)).get("id1").intValue()),(int)(featureLst.get(ruleSatisfier.get(iter)).get("id2").intValue()));
					 
					 if(!queried_edge_map.containsKey(p)) {
						 queries++;
						 queried_edge_map.put(p, goldMap.containsKey(p));
						 System.out.println("queries is2 "+queries);
					 }
					if(goldMap.containsKey(p))
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
		 System.out.println("done here");
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
	public int mergeBlocks() {
		//POpulate ruleBlockSize
		
		for(int jj=1;jj<=2;jj++) {
			System.out.println("came back?");
			Map<Integer, ArrayList<String>> inverted_list = new HashMap<Integer, ArrayList<String>>();
			int depth=1;
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
                String[] aa = tmp.split(";",0);
                //System.out.println(rec_id+" "+aa.length+" "+recordList.get(rec_id));
                		
				if (aa.length==7)
                                tmp = aa[0]+";"+aa[1]+";"+aa[2]+";"+aa[4]+";"+aa[5]+";"+aa[6];
                                else if(aa.length==6)
                                tmp = aa[0]+";"+aa[1]+";"+aa[2]+";"+aa[4]+";"+aa[5];
                                else if(aa.length==5)
                                tmp = aa[0]+";"+aa[1]+";"+aa[2]+";"+aa[4];
				tmp=tmp.replace(";"," ");		
				tmp = tmp.replaceAll("( )+", " ");
				
				String[] rec = tmp.split(" ",0);
				int i=1;
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
				alreadyDone.clear();
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
						
						
						 {
							cleaned_level_map.put(s, level_map.get(s));
							for(int a:level_map.get(s)) {
								
								ArrayList<String> tmp = inverted_list.get(a);
								//System.out.println(a+" "+s);
								tmp.add(s);
								inverted_list.put(a, tmp);
							}
							counter++;
							
							loc+=1;
						}
					}
				}
				if(jj==1) {
					inverted_list1=inverted_list;
					blockMap1 =  cleaned_level_map;
				}
				else {
					inverted_list2=inverted_list;
					blockMap2=( cleaned_level_map);
				}
			}
			
			
			
			tmpBlockTreeMap.clear();	
			System.gc();
			//How do we weigh a pair of records
			System.out.println("done reging"+counter);
		}
		System.out.println("exiting this");
		return 0	;
	}
	
	
	public void processBlocks() throws Exception{
		
		if (mergeBlocks()==0) {
			System.out.println("done merging");
			//return;
		}
		
		//Write the different operators and subroutines here along with features
		//Maintain a global list of queried edges so that I can use them to verify that these are all I know
		int numSample=100000;
		ArrayList<double[]> pairs = samplePairsFalcon(numSample);
		
		ArrayList<double[]> Validation = samplePairsFalcon((int)(0.03*numSample));
		
		System.out.println("pairs "+pairs.size());
		
		ArrayList<pair> green =new ArrayList<pair>(),red=new ArrayList<pair>();
		ArrayList<Integer> AlreadyQueried =  new ArrayList<Integer>();
		
		
		for(int i:recordList1.keySet()) {
			for(int j:recordList2.keySet()) {
				{
					pair p=new pair(i,j);
					if(goldMap.containsKey(p))
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
		//if(queries>0)
			//return;
		//Get unique rules
		ArrayList<HashMap<String,Double[]>> Uniqrulelst = UniqRules( rulelst);
		
		 ArrayList<HashMap<String,Double[]>> finalRules = rulesEval( Uniqrulelst,  featvec,  AlreadyQueried);
		System.out.println(finalRules.size()+"final size"+" "+Uniqrulelst.size()+" "+rulelst.size());
		for(HashMap<String,Double[]> r : finalRules) {
			for(String s:r.keySet())
				System.out.print(s+" "+r.get(s)[1]+" ;;");
			System.out.println(queries+"dsjkl");
		}
		/////Get unique rules
		ArrayList<HashMap<String,Double>> candidatevec = new ArrayList<HashMap<String,Double>>();
		int g=0;
		int rej=0;
		for(int a:recordList1.keySet()) {
			for(int b:recordList2.keySet()) {
				 {pair p = new pair(a,b);
					boolean out = goldMap.containsKey(p);
					HashMap<String,Double> feat = GetFeatures( a,  b,10,out);
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
						if(goldMap.containsKey(p)){
							g++;
							System.out.println(rej+" "+candidatevec.size()+"found green"+g);
						}
					}else {
						rej++;
						;//if(goldMap.get(a).equals(goldMap.get(b)))
							//g++;
					}
				}
				
			}
		}
		System.out.println(queries+" going to train for "+candidatevec.size());
		
		Validation = samplePairsFalcon((int)Math.min(recordList1.keySet().size(),(0.03*candidatevec.size())));
		 randomForest = al_matcher(10,  candidatevec,  AlreadyQueried, V,false); 
		 System.out.println(queries+" "+candidatevec.size()+ " "+V.size()+" "+g);
		
		 g=0;
		 int fg=0;
		for(HashMap<String,Double> feat : candidatevec) {
			int a=feat.get("id1").intValue();
			int b=feat.get("id2").intValue();
			pair p=new pair(a,b);
			if(goldMap.containsKey(p) && queryPair(randomForest,feat)>=0.5 )
				g++;
			else if(!goldMap.containsKey(p) && queryPair(randomForest,feat)>=0.5 )
				fg++;
			
			
			//System.out.println(queryPair(randomForest,feat ));
		}
		System.out.println("greesn is "+g+" "+fg);
		return;
	
	}
	
	
	
	public static void main(String[] args) throws Exception{
    		
    		Falcon2017CleanClean FF = new Falcon2017CleanClean();
  
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
    				attributeList = line.split(";",0);
    				//ArrayList<String> record = new ArrayList<String>();
    				HashMap<String,Integer> recordMap = new HashMap<String,Integer>();
    				int recordId = Integer.parseInt(attributeList[0].trim().replace(".jpg", ""));
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
    			FF.goldMap.put(p1, true);
    		}
    		
    		
    		
    		System.out.println("Memory after goldMap:" + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024+" "+Runtime.getRuntime().freeMemory() / 1024);
    		System.out.println("Going inside process");
    		FF.processBlocks();

    		PrintStream queried = new PrintStream(FF.folder+"/queried.txt");
    		//PrintStream queried = new PrintStream(FF.folder+"/goldgreen.txt");
    		ArrayList<String> featureOrder = new ArrayList<String>();
    		for (pair p:FF.actualqueried_edge_map.keySet()){
    			HashMap<String,Double>  dict= FF.GetFeatures(p.x,p.y,20,FF.actualqueried_edge_map.get(p));
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


    		}

    	}
    	
}

