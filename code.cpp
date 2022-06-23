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

void setQueen(int k, int N, vector<vector<int>>& operations, int& cnt) {
	if (k == N)
	{
		if (operations.size() > 0)
		{ cnt++; }
	}

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
			setQueen(k + 1, N, operations, cnt);
			operations.pop_back();
		}
	}
}

// void traverse(TreeNode* node, vector<int>& vals) {
// 	if (node == nullptr)
// 		return;

// 	traverse(node->left, vals);
// 	traverse(node->right, vals);
// 	vals.push_back(node->val);
// }

vector<int> inorderTraversal(TreeNode* root) {
	vector<int> values;

	traverse(root, values);
	return values;
}

int totalNQueens(int n) {
	int cnt = 0;

	vector<vector<int>> operations;
	setQueen(0, n, operations, cnt);
	return cnt;
}

vector<vector<int>> levelOrder(Node* root) {
	vector<vector<int>> res;
	queue<Node*> curLevel;
	queue<Node*> nextLevel;
	nextLevel.push(root);

	while (true)
	{
		vector<int> values;
		swap(curLevel, nextLevel);
		while (!curLevel.empty())
		{
			Node* node = curLevel.front();

			curLevel.pop();
			if (node)
			{
				values.push_back(node->val);
				for (auto child : node->children)
				{
					if (child)
						nextLevel.push(child);
				}
			}
		}

		if (values.size() > 0)
			res.push_back(values);
		else
			break;
	}

	return res;
}

void traverse(Node* node, vector<int>& res) {
	if (node == nullptr)
		return;

	for (auto& child : node->children)
		traverse(child, res);

	res.push_back(node->val);
}

vector<int> preorder(Node* root) {
	vector<int> res;
	traverse(root, res);

	return res;
}

bool compare(TreeNode* node1, TreeNode* node2) {
	if (node1 == nullptr && node2 != nullptr)
		return false;
	if (node1 != nullptr && node2 == nullptr)
		return false;

	return node1->val == node2->val && compare(node1->left, node2->left) &&
		   compare(node1->right, node2->right);
}

bool isSameTree(TreeNode* node1, TreeNode* node2) {
	if (node1 == node2)
		return true;
	if (node1 == nullptr && node2 != nullptr)
		return false;
	if (node1 != nullptr && node2 == nullptr)
		return false;

	return node1->val == node2->val && isSameTree(node1->left, node2->left) &&
		   isSameTree(node1->right, node2->right);
}

bool isSubtree(TreeNode* root, TreeNode* subRoot) {
	if (root == nullptr && subRoot != nullptr)
		return false;

	if (isSameTree(root, subRoot))
		return true;

	if (isSubtree(root->left, subRoot))
		return true;

	if (isSubtree(root->right, subRoot))
		return true;

	return false;
}

bool isUnivalTree(TreeNode* root) {
	queue<TreeNode*> nodes;
	nodes.push(root);
	int target = root->val;

	while (nodes.empty() == false)
	{
		TreeNode* node = nodes.front();
		nodes.pop();

		if (node->val != target)
			return false;

		if (node->left)
			nodes.push(node->left);

		if (node->right)
			nodes.push(node->right);
	}

	return true;
}

bool isSymmetric(TreeNode* root) {
	queue<TreeNode*> nodes;

	nodes.push(root);
	stack<int> values;
	values.push(root->val);

	int i = 0;
	int cnt = 1;
	int validNum = 0;

	while (nodes.empty() == false)
	{
		TreeNode* node = nodes.front();
		nodes.pop();

		nodes.push(node == nullptr ? nullptr : node->left);
		nodes.push(node == nullptr ? nullptr : node->right);

		if (node && (node->left || node->right))
			validNum++;

		if (i < cnt / 2)
			values.push(node ? node->val : 101);
		else
		{
			if (node == nullptr && values.top() != 101)
				return false;

			if (node && node->val != values.top())
				return false;

			values.pop();
		}

		i++;
		if (i >= cnt)
		{
			i = 0;
			cnt *= 2;

			if (validNum == 0)
				break;

			validNum = 0;
		}
	}

	return true;
}

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	int i = m - 1, j = n - 1, k = m + n - 1;
	for (int k = m + n - 1; i >= 0 && j >= 0; k--)
	{
		nums1[k] = max(nums1[i], nums2[j]);
		if (nums1[i] < nums2[j])
			j--;
		else
			i--;
	}

	while (i >= 0)
	{
		nums1[k] = nums1[i];
		i--;
		k--;
	}

	while (j >= 0)
	{
		nums1[k] = nums2[j];
		j--;
		k--;
	}
}

int minDepth(TreeNode* root) {
	queue<TreeNode*> curLevel;
	queue<TreeNode*> nextLevel;

	int depth = 0;
	if (root)
		curLevel.push(root);

	while (curLevel.empty() == false)
	{
		TreeNode* node = curLevel.front();
		curLevel.pop();

		if (node && node->left)
			nextLevel.push(node->left);
		if (node && node->right)
			nextLevel.push(node->right);

		if (node && node->left == nullptr && node->right == nullptr)
		{
			depth++;
			break;
		}

		if (curLevel.empty())
		{
			depth++;
			swap(curLevel, nextLevel);
		}
	}

	return depth;
}

int lengthOfLongestSubstring(string s) {
	int begin = 0;

	vector<int> chars(26, -1);

	int maxI = 0;
	int maxL = 0;

	int i = 0;
	for (; i < s.length(); ++i)
	{
		int prev = chars[s[i] - 'a'];
		if (prev >= 0 && prev >= begin)
		{
			// appear a same character
			int cnt = i - begin;
			if (maxL < cnt)
			{
				maxL = cnt;
				maxI = begin;
			}

			begin = prev + 1;
		}

		chars[s[i] - 'a'] = i;
	}
	maxL = max(maxL, i - begin);

	return maxL;
}

vector<int> twoSum(vector<int>& numbers, int target) {
	vector<int> nums(2001, -1);

	vector<int> res;
	for (int i = 0; i < numbers.size(); ++i)
	{
		int tag = target - numbers[i] + 1000;
		if (nums[tag] != -1)
		{
			res.push_back(nums[tag] + 1);
			res.push_back(i + 1);
			break;
		}
		else
		{ nums[numbers[i] + 1000 + 1] = i; }
	}

	return res;
}

void traverse(TreeNode* node, vector<vector<int>>& res, int depth) {
	if (node == nullptr)
		return;

	if (res.size() <= depth)
		res.push_back(vector<int>());

	traverse(node->left, res, depth + 1);
	traverse(node->right, res, depth + 1);

	res[depth].push_back(node->val);
}

vector<vector<int>> levelOrderBottom(TreeNode* root) {
	vector<vector<int>> res;

	res.push_back(vector<int>());
	traverse(root, res, 0);

	reverse(res.begin(), res.end());
	return res;
}

bool leafSimilar(TreeNode* root1, TreeNode* root2) {
	vector<int> leaves;
	deque<TreeNode*> nodes;

	nodes.push_back(root1);
	while (!nodes.empty())
	{
		TreeNode* node = nodes.front();
		nodes.pop_front();
		if (node)
		{
			if (node->left == nullptr && node->right == nullptr)
				leaves.push_back(node->val);
			else
			{
				if (node->right)
					nodes.push_front(node->right);
				if (node->left)
					nodes.push_front(node->left);
			}
		}
	}

	int idx = 0;
	nodes.clear();
	nodes.push_back(root2);
	while (!nodes.empty())
	{
		TreeNode* node = nodes.front();
		nodes.pop_front();
		if (node)
		{
			if (node->left == nullptr && node->right == nullptr)
			{
				if (node->val != leaves[idx])
					return false;
				idx++;
			}
			else
			{
				if (node->right)
					nodes.push_front(node->right);
				if (node->left)
					nodes.push_front(node->left);
			}
		}
	}

	return idx >= leaves.size();
}

int minOperations(vector<int>& nums, int x) {
	int total = accumulate(nums.begin(), nums.end(), 0);

	if (total < x)
		return -1;
	if (total == x)
		return nums.size();

	int left = 0;
	int current = 0;
	int step = -1;

	for (int right = 0; right < nums.size(); ++right)
	{
		current += nums[right];
		while (current > total - x && left <= right)
		{
			current -= nums[left];
			left++;
		}

		if (current == total - x)
			step = max(step, right - left + 1);
	}

	return step != -1 ? nums.size() - step : -1;
}

int minSubArrayLen(int target, vector<int>& nums) {
	int left = 0, right = 0;
	int current = 0;
	int len = nums.size();

	if (accumulate(nums.begin(), nums.end(), 0) == target)
		return len;

	for (; right < nums.size(); ++right)
	{
		current += nums[right];
		while (current > target && left <= right)
		{
			current -= nums[left];
			++left;
		}

		if (current < target && left > 0)
		{
			--left;
			current += nums[left];
		}

		len = min(len, right - left + 1);
	}

	return len != nums.size() ? len : 0;
}

TreeNode* trimBST(TreeNode* root, int low, int high) {
	if (root == nullptr)
		return nullptr;

	TreeNode* cur = (root->val >= low && root->val <= high) ? root : nullptr;
	TreeNode* lchild = trimBST(root->left, low, high);
	TreeNode* rchild = trimBST(root->right, low, high);

	if (cur)
	{
		cur->left = lchild;
		cur->right = rchild;
	}
	else
	{
		if (lchild && rchild)
		{
			cur = lchild;
			cur->right = rchild;
		}
		else
		{ cur = lchild ? lchild : (rchild ? rchild : nullptr); }
	}

	return cur;
}

TreeNode* removeLeafNodes(TreeNode* root, int target) {
	if (root == nullptr)
		return nullptr;

	TreeNode* lchild = removeLeafNodes(root->left, target);
	TreeNode* rchild = removeLeafNodes(root->right, target);

	root->left = lchild;
	root->right = rchild;

	if (lchild == nullptr && rchild == nullptr && root->val == target)
		root = nullptr;

	return root;
}

// prefix tree
class Trie {
public:
	Trie() : root(new TrieNode) {}

	~Trie() { delete root; }

	void insert(string word) {
		TrieNode* p = root;
		for (const char c : word)
		{
			if (!p->children.count(c))
				p->children[c] = new TrieNode();
			p = p->children[c];
		}
		p->isWord = true;
	}

	bool search(string word) {
		const TrieNode* p = find(word);
		return p && p->isWord;
	}

	bool startsWith(string prefix) { return find(prefix) != nullptr; }

private:
	struct TrieNode {
		unordered_map<char, TrieNode*> children;
		bool isWord;

		TrieNode() : isWord(false) {}

		~TrieNode() {
			for (auto& child : children)
			{
				if (child.second)
					delete child.second;
			}
		}
	};

	const TrieNode* find(const string& prefix) const {
		const TrieNode* p = root;
		for (const char c : prefix)
		{
			if (!p->children.count(c))
				return nullptr;

			p = p->children.at(c);
		}

		return p;
	}

	TrieNode* root;
};

int traverse(vector<int>& nums, int left, int right) {
	if (left + 1 >= right)
		return min(nums[left], nums[right]);

	if (nums[left] < nums[right])
		return nums[left];

	int mid = left + (right - left) / 2;
	return min(traverse(nums, left, mid), traverse(nums, mid + 1, right));
}

int findMin(vector<int>& nums) {
	return traverse(nums, 0, nums.size() - 1);
}

bool hasPathSum(TreeNode* root, int targetSum) {
	if (root == nullptr)
		return false;

	if (root->left == nullptr && root->right == nullptr)
		return targetSum == root->val;

	bool left =
		root->left ? hasPathSum(root->left, targetSum - root->val) : false;
	bool right =
		root->right ? hasPathSum(root->right, targetSum - root->val) : false;

	return left || right;
}

int numOfPaths(TreeNode* node, int target) {
	if (node == nullptr)
		return 0;
	target -= node->val;

	return (target == 0 ? 1 : 0) + numOfPaths(node->left, target) +
		   numOfPaths(node->right, target);
}

int pathSum(TreeNode* root, int targetSum) {
	if (root == nullptr)
		return 0;
	return numOfPaths(root, targetSum) + pathSum(root->left, targetSum) +
		   pathSum(root->right, targetSum);
}

int subarraySum(vector<int>& nums, int k) {
	map<int, int> dp;

	int curSum = 0;
	int cnt = 0;
	for (auto num : nums)
	{
		curSum += num;
		if (dp.count(curSum - k))
			cnt += dp[curSum - k];

		if (curSum == k)
			cnt += 1;

		dp[curSum] = dp.count(curSum) ? dp[curSum] + 1 : 1;
	}

	return cnt;
}

bool checkSubarraySum(vector<int> nums, int k) {
	map<int, vector<int>> dp;

	int curSum = 0;

	for (int i = 0; i < nums.size(); ++i)
	{
		curSum += nums[i];
		if (curSum % k == 0 && i >= 1)
			return true;

		if (i >= 1 && nums[i] == 0 && nums[i - 1] == 0)
			return true;

		int rat = 1;
		while (k * rat <= curSum)
		{
			if (dp.count(curSum - k * rat))
			{
				vector<int> idx = dp[curSum - k * rat];
				for (auto id : idx)
				{
					if (i - id >= 1)
						return true;
				}
			}
			++rat;
		}

		if (dp.count(curSum))
			dp[curSum].push_back(i);
		else
			dp[curSum] = {i};
	}

	return false;
}

int numSubarrayProductLessThanK(vector<int>& nums, int k) {
	int cnt = 0;

	int left = 0, right = 0;
	int cur = 1;

	for (; right < nums.size() && left <= right;)
	{
		cur *= nums[right];

		while (cur >= k && left <= right)
		{
			cur /= nums[left];
			++left;
		}

		if (cur < k && right >= left)
			cnt += right - left + 1;

		right++;
	}

	return cnt;
}

int subarraysDivByK(vector<int>& nums, int k) {
	unordered_map<int, int> dp;
	int cnt = 0;
	int sum = 0;

	for (auto num : nums)
	{
		sum += num;
		sum = sum % k;
		sum = sum < 0 ? sum + k : sum;

		if (dp.count(sum))
			cnt += dp[sum];

		if (sum == 0)
			++cnt;

		dp[sum] = dp.count(sum) ? dp[sum] + 1 : 1;
	}

	return cnt;
}

int rob(vector<int>& nums) {
	int n = nums.size();
	vector<int> dp(n + 1, 0);
	dp[1] = nums[0];

	for (int i = 2; i <= n; ++i)
	{ dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1]); }

	return dp[n];
}

int traverse(unordered_map<int, int>& memo, vector<int>& cost, int level) {
	if (level <= 1)
		return 0;

	if (memo.count(level))
		return memo[level];

	memo[level] = min(traverse(memo, cost, level - 1) + cost[level - 1],
					  traverse(memo, cost, level - 2) + cost[level - 2]);
	return memo[level];
}

int minCostClimbingStairs(vector<int>& cost) {
	unordered_map<int, int> memo;
	return traverse(memo, cost, cost.size());
}

int minCost(vector<vector<int>>& costs) {
	int R = costs[0][0], G = costs[0][1], B = costs[0][2];

	for (int i = 1; i < costs.size(); ++i)
	{
		int r = costs[i][0] + min(G, B);
		int g = costs[i][1] + min(R, B);
		int b = costs[i][2] + min(R, G);
		R = r;
		G = g;
		B = b;
	}

	return min(R, min(G, B));
}

int tribonacci(int n) {
	int T[3] = {0, 1, 1};
	if (n <= 2)
		return T[n];

	for (int i = 3; i <= n; ++i)
	{
		int tmp = T[0] + T[1] + T[2];
		T[0] = T[1];
		T[1] = T[2];
		T[2] = tmp;
	}
	return T[2];
}

int getMax(vector<int>& nums, unordered_map<int, int>& dp, int r) {
	if (r < 0)
		return 0;

	int cnt = 0;

	int cur = nums[r];
	while (r >= 0 && nums[r] == cur)
	{
		--r;
		++cnt;
	}

	// get next in-adjacent node
	int next = r;
	while (next >= 0 && nums[next] == cur - 1)
	{ --next; }

	int selected = cnt * cur + getMax(nums, dp, next);
	if (next == r)	// always in-adjacent
		dp[r + cnt] = selected;
	else
		dp[r + cnt] = max(selected, getMax(nums, dp, r));

	return dp[r + cnt];
}

int deleteAndEarn(vector<int> nums) {
	unordered_map<int, int> points;
	vector<int> uniqueNums;
	for (auto num : nums)
	{
		if (!points.count(num))
			uniqueNums.push_back(num);
		points[num] += num;
	}

	sort(uniqueNums.begin(), uniqueNums.end());

	int oneBack = points[uniqueNums[0]];
	int twoBack = 0;

	for (int i = 1; i < uniqueNums.size(); ++i)
	{
		int tmp = oneBack;
		if (uniqueNums[i] == uniqueNums[i - 1] + 1)
		{ oneBack = max(oneBack, twoBack + points[uniqueNums[i]]); }
		else
		{ oneBack += points[uniqueNums[i]]; }
		twoBack = tmp;
	}

	return oneBack;
}

int dp(vector<int>& multipliers,
	   vector<int>& nums,
	   vector<vector<int>>& memo,
	   int left,
	   int right,
	   int i) {
	int m = multipliers.size();

	if (i == m)
		return 0;

	int mult = multipliers[i];

	if (memo[i][left] == 0)
	{
		memo[i][left] = max(mult * nums[left] + dp(multipliers, nums, memo,
												   left + 1, right, i + 1),
							mult * nums[right] + dp(multipliers, nums, memo,
													left, right - 1, i + 1));
	}

	return memo[i][left];
}

int maximumScore(vector<int>& nums, vector<int>& multipliers) {
	int m = multipliers.size();
	vector<vector<int>> memo(m + 1, vector<int>(m + 1, 0));

	for (int i = m - 1; i >= 0; --i)
	{
		for (int left = i; left >= 0; --left)
		{
			int right = nums.size() - 1 - (i - left);
			memo[i][left] =
				max(multipliers[i] * nums[left] + memo[i + 1][left + 1],
					multipliers[i] * nums[right] + memo[i + 1][left]);
		}
	}

	return memo[0][0];
	// Time Limit Exceeded: Top-down
	// return dp(multipliers, nums, memo, 0, nums.size() - 1, 0);
}

int longestCommonSubsequence(string text1, string text2) {
	int m = text1.length(), n = text2.length();
	vector<vector<int>> memo(m, vector<int>(n, 0));
	memo[0][0] = text1[0] == text2[0] ? 1 : 0;

	int prev = memo[0][0];
	for (int x = 1; x < m; ++x)
	{
		if (text1[x] == text2[0])
			prev = 1;
		memo[x][0] = prev;
	}
	prev = memo[0][0];
	for (int y = 1; y < n; ++y)
	{
		if (text2[y] == text1[0])
			prev = 1;
		memo[0][y] = prev;
	}

	for (int i = 1; i < m; ++i)
	{
		for (int j = 1; j < n; ++j)
		{
			if (text1[i] == text2[j])
				memo[i][j] = memo[i - 1][j - 1] + 1;
			else
				memo[i][j] = max(memo[i][j - 1], memo[i - 1][j]);
		}
	}

	return memo[m - 1][n - 1];
}

int maximalSquare(vector<vector<char>>& matrix) {
	int m = matrix.size(), n = matrix[0].size();

	vector<vector<int>> memo(m + 1, vector<int>(n + 1, 0));
	int cnt = 0;
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (matrix[i - 1][j - 1] == '1')
				memo[i][j] = min(memo[i - 1][j - 1],
								 min(memo[i][j - 1], memo[i - 1][j])) +
							 1;
			else
				memo[i][j] = 0;

			cnt = max(cnt, memo[i][j]);
		}
	}

	return cnt * cnt;
}

class PriorityQueue {
public:
	PriorityQueue(int _max) : maxSize(_max), size(0) {}

private:
	vector<int> nums;
	int maxSize;
	int size;

	int GetParent(int id) { return (id - 1) / 2; }

	void BuildUp() {
		int id = size - 1;

		while (id > 0)
		{
			int parent = (id - 1) / 2;
			int brother = (id % 2 == 0) ? id - 1 : id + 1;
			brother = brother >= size ? size - 1 : brother;

			if (nums[id] <= nums[brother] && nums[id] < nums[parent])
				swap(nums[id], nums[parent]);
			else if (nums[brother] <= nums[id] && nums[brother] < nums[parent])
				swap(nums[brother], nums[parent]);
			else
				break;

			id = parent;
		}
	}

	void BuildDown() {
		int id = 0;
		while (id < size)
		{
			int left = id * 2 + 1;
			left = min(left, size - 1);
			int right = left + 1;
			right = min(right, size - 1);

			if (nums[id] > nums[left] && nums[right] >= nums[left])
			{
				swap(nums[id], nums[left]);
				id = left;
			}
			else if (nums[id] > nums[right] && nums[left] >= nums[right])
			{
				swap(nums[id], nums[right]);
				id = right;
			}
			else
				break;
		}
	}

public:
	void Push(int val) {
		if (maxSize <= 0 || size < maxSize)
		{
			++size;
			nums.push_back(val);

			BuildUp();
		}
		else if (maxSize > 0 && size == maxSize)
		{
			if (val > nums[0])
			{
				nums[0] = val;
				BuildDown();
			}
		}

		Display();
	}

	int Get() {
		// assert(size > 0);
		swap(nums[0], nums[size - 1]);
		--size;

		int val = nums[size];
		nums.pop_back();
		return val;
	}

	void Display() {
		for (auto num : nums)
			cout << num << " ";
		cout << endl;
	}
};

int main() {
	PriorityQueue mQueue(2);
	vector<int> nums{3, 2, 1, 5, 6, 4};

	for (auto num : nums)
	{ mQueue.Push(num); }

	cout << mQueue.Get();
	return 0;
}