

import java.util.ArrayList;

public class component {
	private  ArrayList<Integer> nodes = new ArrayList<Integer>();;
	public component(ArrayList<Integer> init_nodes){
		nodes =  init_nodes;
	}
	public component(){
		
	}
	public ArrayList<Integer> get_component(){
		return nodes;
	}
	
}
