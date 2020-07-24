#include <iostream>
class Solution{
public:
vector<vector<int>> res;
vector<vector<int>> permute(vector<int>&nums){
    vector<int> path;
    backtrack(nums, path);
    return res;
}
}