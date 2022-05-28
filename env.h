#ifndef _ENV_H
#define _ENV_H

#include <vector>
#include <set>
#include <map>
#include <list>
#include <queue>
#include <memory>
#include <algorithm>
#include <limits.h>
#include <cstddef>
#include <algorithm>
#include <typeinfo>
#include <iostream>

#include <math.h>
#include <numeric>
#include <unordered_map>
#include <stack>
#include <unordered_set>

#include <thread>
#include <mutex>

#include <time.h>
#include <utility>
#include <bitset>

#include <nlohmann/json.hpp>

using namespace std;

void display_vector(const vector<int>& nums) {
	for (int i = 0; i < nums.size(); ++i)
	{ cout << nums[i] << " "; }
	cout << endl;
}

template <typename T>
void print_vector(vector<T> array) {
	cout << "[";
	for (int i = 0; i < array.size(); ++i)
	{
		cout << array[i];
		if (i < array.size() - 1)
			cout << ", ";
	}
	cout << "]" << endl;
}

vector<int> direction{-1, 0, 1, 0, -1};

// Definition for a binary tree node.
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right)
		: val(x), left(left), right(right) {}
};

#endif