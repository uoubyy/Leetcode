#include "env.h"

int backtracing(int n, map<int, int>& steps) {
	if (n <= 0)
		return 0;
	if (steps.count(n))
		return steps[n];

	int num = backtracing(n - 1, steps);
	num += backtracing(n - 2, steps);
	steps[n] = num;
	return num;
}

int climbStairs(int n) {
	map<int, int> steps;
	steps[1] = 1;
	steps[2] = 2;

	return backtracing(n, steps);
}

int getMiniRounds(int n, map<int, int>& steps) {
	if (n < 2)
		return -1;
	if (steps.count(n))
		return steps[n];

	int ans = n;
	if (getMiniRounds(n - 2, steps) > 0)
		ans = min(ans, getMiniRounds(n - 2, steps) + 1);
	if (getMiniRounds(n - 3, steps) > 0)
		ans = min(ans, getMiniRounds(n - 3, steps) + 1);

	ans = (ans == n ? -1 : ans);
	steps[n] = ans;

	return ans;
}

int minimumRounds(vector<int> tasks) {
	map<int, int> taskInfo;
	for (auto task : tasks)
	{
		if (taskInfo.count(task))
			taskInfo[task]++;
		else
			taskInfo[task] = 1;
	}

	int cnt = 0;

	map<int, int> steps;
	steps[2] = 1;
	steps[3] = 1;

	for (auto task : taskInfo)
	{
		if (task.second == 1)
			return -1;

		if (getMiniRounds(task.second, steps) <= 0)
			return -1;

		cnt += getMiniRounds(task.second, steps);
	}

	return cnt;
}

vector<string> readBinaryWatch(int turnedOn) {
	map<int, vector<string>> times;
	for (unsigned int h = 0; h < 12; ++h)
	{
		for (unsigned int m = 0; m < 60; ++m)
		{
			bitset<4> hours(h);
			bitset<6> minutes(m);

			char buffer[10];
			sprintf(buffer, "%d:%02d", hours, minutes);
			times[hours.count() + minutes.count()].push_back(buffer);
		}
	}
	return times[turnedOn];
}

int maxInChild(TreeNode* node, int& curMax) {
	if (node == nullptr)
		return INT_MIN;

	int nodeMax = node->val;

	if (node->left == nullptr && node->right == nullptr)
		return node->val;

	int l = maxInChild(node->left, curMax);
	int r = maxInChild(node->right, curMax);

	bool l_valid = node->left && l > INT_MIN;
	bool r_valid = node->right && r > INT_MIN;

	if (l_valid && r_valid)
		nodeMax = max(
			l,
			max(r, max(node->val, max(node->val + l,
									  max(node->val + r, node->val + l + r)))));
	else if (l_valid)
		nodeMax = max(l, max(node->val, node->val + l));
	else
		nodeMax = max(r, max(node->val, node->val + r));

	curMax = max(curMax, nodeMax);

	if (l_valid && r_valid)
		return max(l + node->val, max(r + node->val, node->val));
	else if (l_valid)
		return max(l + node->val, node->val);
	else
		return max(r + node->val, node->val);

	return node->val;
}

int maxPathSum(TreeNode* root) {
	int curMax = root->val;
	int value = maxInChild(root, curMax);
	return max(curMax, value);
}

int getNext(int i, int n) {
	return (i + 1) % n;
}

int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int l = gas.size() - 1, r = 0, sum = gas[l] - cost[l];
	while (l != r)
	{
		int i = sum < 0 ? --l : r++;
		sum += gas[i] - cost[i];
	}
	return sum >= 0 ? l : -1;
}

string largestNumber(vector<int>& nums) {
	vector<vector<string>> cnt(10);

	for (int i = 0; i < nums.size(); ++i)
	{
		int num = nums[i];
		while (num >= 10)
			num /= 10;

		cnt[num].push_back(to_string(nums[i]));
	}

	string ans = "";
	for (int i = 9; i >= 0; --i)
	{
		if (cnt[i].size() > 0)
		{
			sort(cnt[i].begin(), cnt[i].end(),
				 [](string a, string b) { return (a + b) > (b + a); });

			for (auto& str : cnt[i])
				ans += str;
		}
	}

	if (ans[0] == '0')
		return "0";

	return ans;
}

int minMeetingRooms(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end(),
		 [](vector<int>& a, vector<int>& b) { return a[0] < b[0]; });

	vector<int> rooms;
	rooms.push_back(intervals[0][1]);
	for (int i = 1; i < intervals.size(); ++i)
	{
		bool find = false;
		for (int j = 0; j < rooms.size(); ++j)
		{
			if (intervals[i][0] >= rooms[j])
			{
				rooms[j] = intervals[i][1];
				find = true;
				break;
			}
		}

		if (!find)
			rooms.push_back(intervals[i][1]);
	}

	return rooms.size();
}

bool knows(int a, int b) {
	return false;
}

int findCelebrity(int n) {
	int l = 0, r = 1;
	int host = l;
	while (l < n && r < n)
	{
		if (knows(l, r))
		{
			host = r;
			l = r;
			r++;
		}
		else
		{
			host = l;
			r++;
		}
	}

	for (int i = 0; i < host; ++i)
	{
		if (!knows(i, host))
			return -1;
		if (knows(host, i))
			return -1;
	}

	for (int i = host + 1; i < n; ++i)
	{
		if (!knows(i, host))
			return -1;
		if (knows(host, i))
			return -1;
	}

	return host;
}

// 997
int findJudge(int n, vector<vector<int>>& trust) {
	vector<int> cnt(n + 1, 0);

	for (auto& info : trust)
	{
		cnt[info[0]] = -1;
		if (cnt[info[1]] >= 0)
			cnt[info[1]]++;
	}

	int ans = -1;
	for (int i = 1; i <= n; ++i)
	{
		if (cnt[i] == n - 1)
			ans = i;
	}

	return ans;
}

// 280
void wiggleSort(vector<int>& nums) {
	bool lessEqual = true;

	for (int i = 0; i < nums.size() - 1; ++i)
	{
		if (lessEqual && nums[i] > nums[i + 1])
			swap(nums[i], nums[i + 1]);

		if (!lessEqual && nums[i] < nums[i + 1])
			swap(nums[i], nums[i + 1]);
		lessEqual = !lessEqual;
	}
}

void traverse(TreeNode* node, int& maxD, int depth) {
	if (node == nullptr)
		return;

	depth++;
	maxD = max(maxD, depth);
	traverse(node->left, maxD, depth);
	traverse(node->right, maxD, depth);
}

int maxDepth(TreeNode* root) {
	int maxD = 0;
	traverse(root, maxD, 0);
	return maxD;
}

int treeDepth(TreeNode* node) {
	if (node == nullptr)
		return 0;
	int lh = treeDepth(node->left);
	int rh = treeDepth(node->right);

	if (lh == -1 || rh == -1)  // child tree nnot balanced
		return -1;

	if (lh - rh > 1 || lh - rh < -1)
		return -1;

	return max(lh, rh) + 1;
}

bool isBalanced(TreeNode* root) {
	return treeDepth(root) != -1;
}

void traverse(TreeNode* node, vector<int>& elements) {
	if (node == nullptr)
		return;
	traverse(node->left, elements);
	elements.push_back(node->val);
	traverse(node->right, elements);
}

void mergeTrees(TreeNode* root1, TreeNode* root2, vector<int>& elements) {
	if (root1 == nullptr)
	{
		traverse(root2, elements);
		return;
	}
	else if (root2 == nullptr)
	{
		traverse(root1, elements);
		return;
	}

	TreeNode* minNode = root1->left;
	if (minNode == nullptr)
		minNode = root2->left;
	else if (root2->left)
		minNode = root1->left->val < root2->left->val ? minNode : root2->left;

	traverse(minNode, elements);

	if (minNode == root2->left)
	{
		if (root1->left)
		{
			if (root1->left->val >= root2->val)
				elements.push_back(root2->val);
			root2 = root2->right;
		}
	}

	if (minNode == root2->left && root1->left && root1->left->val >= root2->val)
	{
		// collapse root2
		// root2.left must be mini node
		elements.push_back(root2->val);
		root2 = root2->right;
	}
	else if (root2->left && root2->left->val >= root1->val)
	{
		// root1.left must be mini node
		elements.push_back(root1->val);
		root1 = root1->right;
	}
	else
	{
		if (minNode != root1->left)
			traverse(root1->left, elements);
		if (minNode != root2->left)
			traverse(root2->left, elements);
		if (root1->val < root2->val)
		{
			elements.push_back(root1->val);
			root1 = root1->right;
		}
		else
		{
			elements.push_back(root2->val);
			root2 = root2->right;
		}
	}

	mergeTrees(root1, root2, elements);
}

vector<int> getAllElements(TreeNode* root1, TreeNode* root2) {
	vector<int> allElements;
	mergeTrees(root1, root2, allElements);
	return allElements;
}

vector<vector<int>> transpose(vector<vector<int>>& matrix) {
	int m = matrix.size();
	int n = matrix[0].size();

	vector<vector<int>> matT(n, vector<int>(m, 0));

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
			matT[i][j] = matrix[j][i];
	}
	return matT;
}

int wiggleMaxLength(vector<int> nums) {
	int n = nums.size();
	if (n == 1)
		return 1;

	int maxLength = nums[0] == nums[1] ? 1 : 2;
	int prevDiff = nums[1] - nums[0];

	for (int i = 2; i < n; ++i)
	{
		int diff = nums[i] - nums[i - 1];
		if (diff * prevDiff < 0)
		{
			maxLength++;
			prevDiff = diff;
		}
		else if (prevDiff == 0 && diff != 0)
		{
			maxLength++;
			prevDiff = diff;
		}
	}

	return maxLength;
}

int traverse(int n, map<int, int>& dp) {
	if (n == INT_MAX)
		return traverse(n - 1, dp);

	if (dp.count(n))
		return dp[n];

	if (n % 2 == 0)
	{
		int depth = traverse(n / 2, dp);
		dp[n] = depth + 1;
	}
	else
	{
		int d1 = traverse(n + 1, dp);
		int d2 = traverse(n - 1, dp);
		dp[n] = min(d1, d2) + 1;
	}

	return dp[n];
}

int integerReplacement(int n) {
	map<int, int> dp;
	dp[1] = 0;
	dp[2] = 1;

	return traverse(n, dp);
}

vector<int> runningSum(vector<int>& nums) {
	vector<int> sums;
	sums.push_back(nums[0]);

	for (int i = 1; i < nums.size(); ++i)
		sums.push_back(nums[i] + sums[i - 1]);

	return sums;
}

string removeKdigits(string num, int k) {
	if (num.length() <= k)
		return "0";

	stack<char> res;
	res.push(num[0]);

	int i = 1;
	for (; i < num.length(); ++i)
	{
		while (k > 0 && !res.empty() && num[i] < res.top())
		{
			res.pop();
			k--;
		}
		res.push(num[i]);

		if (k <= 0)
			break;
	}

	string number;
	while (!res.empty())
	{
		number.push_back(res.top());
		res.pop();
	}

	reverse(number.begin(), number.end());
	if (i < num.length())
		number += num.substr(i + 1);

	number = number.substr(0, number.length() - k);
	i = 0;
	for (; i < number.length() - 1; ++i)
	{
		if (number[i] != '0')
			break;
	}

	return number.substr(i);
}

int longestPalindrome(string s) {
	map<char, int> nums;

	for (auto c : s)
	{ nums[c]++; }

	int len = 0;

	int extra = 0;
	for (auto itor = nums.begin(); itor != nums.end(); itor++)
	{
		if (itor->second % 2 == 0)
			len += itor->second;
		else
		{
			len += itor->second - 1;
			extra = 1;
		}
	}

	return len + extra;
}

class NumMatrix {
public:
	NumMatrix(vector<vector<int>>& matrix) {
		m = matrix.size();
		n = matrix[0].size();
		sumMat = vector<vector<int>>(m + 1, vector<int>(n + 1, 0));

		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				sumMat[i + 1][j + 1] = sumMat[i][j + 1] + sumMat[i + 1][j] +
									   matrix[i][j] - sumMat[i][j];
			}
		}
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return sumMat[row2 + 1][col2 + 1] - sumMat[row2 + 1][col1] -
			   sumMat[row1][col2 + 1] + sumMat[row1][col1];
	}

private:
	vector<vector<int>> sumMat;
	int m;
	int n;
};

int minProductSum(vector<int>& nums1, vector<int>& nums2) {
	sort(nums1.begin(), nums1.end(), greater<int>());
	sort(nums2.begin(), nums2.end());

	int sum = 0;
	for (int i = 0; i < nums1.size(); ++i)
		sum += nums1[i] * nums2[i];

	return sum;
}

int minPartitions(string n) {
	int cnt = 1;
	for (int i = 0; i < n.length(); ++i)
		cnt = max(cnt, n[i] - '0');

	return cnt;
}

bool isVaild(int x0, int y0, int x1, int y1) {
	if (x0 == x1 || y0 == y1)
		return false;
	if (abs(x0 - x1) == abs(y0 - y1))
		return false;

	return true;
}

void setQueen(int k,
			  int N,
			  vector<vector<int>>& operations,
			  vector<vector<string>>& res) {
	if (k == N)
	{
		if (operations.size() > 0)
		{
			vector<string> solution;
			for (auto& op : operations)
			{
				string row(N, '.');
				row[op[1]] = 'Q';
				solution.push_back(row);
			}
			res.push_back(solution);
		}
	}

	printf("start set queen %d\n", k);

	for (int i = 0; i < N; ++i)	 // try to set the queen in each col
	{
		if (k == 0)
			operations.clear();

		bool valid = true;
		for (auto& op : operations)
		{
			if (op[1] == i || (abs(op[1] - i) == abs(op[0] - k)))
			{
				valid = false;
				break;
			}
		}

		if (valid)
		{
			operations.push_back({k, i});
			setQueen(k + 1, N, operations, res);
			operations.pop_back();
		}
	}
}

vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> ans;

	vector<vector<int>> operations;
	setQueen(0, n, operations, ans);
	return ans;
}

int main() {
	vector<string> ans = readBinaryWatch(2);
	// print_vector(ans);

	cout << integerReplacement(8) << endl;
	cout << INT_MAX << endl;
	cout << removeKdigits("1432219", 3) << endl;

	solveNQueens(5);

	return 0;
}