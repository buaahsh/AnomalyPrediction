public class Test {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String str1 = "aaacaaa";
		String str2 = "aca";
		String str3 = "aa";
		System.out.println(func(str1, str2, str3));
	}
	
	public static String func(String str1, String str2, String str3) {
		boolean b1 = Output(str1, str2, str3);
		boolean b2 = Output(reverse1(str1) , str2, str3);
		if (b1 && b2)
		{
			return "both";
		}
		else if (b1) {
			return "forward";
		}
		else if (b2) {
			return "backward";
		}
		return "invalid";
	}
	
	public static boolean Output(String str1, String str2, String str3){
		int start = 0;
		int t = str1.indexOf(str2, start);
		if (t == -1)
			return false;
		start = t + 1;
		t = str1.indexOf(str3, start);
		if (t == -1)
			return false;
		return true;
	}
	
	public static String reverse1(String str)
	{
	   return new StringBuffer(str).reverse().toString();
	}
}
