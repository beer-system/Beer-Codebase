

public class pair {
	public int x,y;

	public pair (int x, int y) {
        this.x = x;
        this.y = y;
    }
	
	public boolean equals(Object o) {
        if (this == o) return true;
       // if (!(o instanceof pair)) return false;
        pair key = (pair) o;
        return (x == key.x && y == key.y)||(x == key.y && y == key.x);
    }
	public int hashCode() {
        int result = x;
        result = 31 * result + y;
        return result;
    }

}
