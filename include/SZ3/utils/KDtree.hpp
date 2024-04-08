#ifndef SZ3_KDTREE_HPP
#define SZ3_KDTREE_HPP

#endif //SZ3_KDTREE_HPP

#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/utils/ByteUtil.hpp"
#include <vector>
#include <algorithm>
#include <stack>

namespace SZ3 {

namespace KDNodeSpace{
    int k;
}

template<class T>
class KDNode{
public:

    int l=-1,r=-1; // left and right son
    std::vector<T> pos;
    T L[3],R[3]; // boundaries

    // pre-quantized points
    std::vector<int64_t> q;

    bool operator<(const KDNode<T> B) const{
//        printf("k = %d\n",k);
//        return pos[k] < b.pos[k];
        return pos[KDNodeSpace::k] < B.pos[KDNodeSpace::k];
    }
};

template<class T>
class KDTree{
public:

    int root=-1;
    // the number of nodes and the number of nodes after decoration
    size_t n=-1,nad=-1;
    KDNode<T>* tree;

//    T eb;
    int64_t bx,by,bz;
//    T lx,ly,lz;

    KDTree(int64_t bx, int64_t by, int64_t bz) :
            bx(bx), by(by), bz(bz) {
    }

    void clear(){
        root=n=nad=0;
        bx=by=bz=0;
        delete[] tree;
    }

    void pushup(int p){

//        printf("%d %d %d\n",p,tree[p].l,tree[p].r);

        for(int i=0;i<3;i++){
            tree[p].L[i] = tree[p].R[i] = tree[p].pos[i];
            if(~tree[p].l){
                tree[p].L[i] = std::min(tree[p].L[i],tree[tree[p].l].L[i]);
                tree[p].R[i] = std::max(tree[p].R[i],tree[tree[p].l].R[i]);
            }
            if(~tree[p].r){
                tree[p].L[i] = std::min(tree[p].L[i],tree[tree[p].r].L[i]);
                tree[p].R[i] = std::max(tree[p].R[i],tree[tree[p].r].R[i]);
            }
        }
    }

    int build(int l, int r, int k_){

        if(l>=r) return -1;
        int mid=(l+r)/2;

        KDNodeSpace::k = k_;
        std::nth_element(tree+l,tree+mid,tree+r);

        tree[mid].l = build(l,mid,(k_+1)%3);
        tree[mid].r = build(mid+1,r,(k_+1)%3);

        pushup(mid);
        return mid;
    }

    void build(std::vector<std::vector<T>>& data){
        n = data.size();
        tree = new KDNode<T>[n];
        for(int i=0;i<n;i++) tree[i].pos = data[i];
        root = build(0,n,0);
    }

    void build(std::vector<T> *data, size_t data_size){
        n = data_size;
        tree = new KDNode<T>[n];
        for(int i=0;i<n;i++) tree[i].pos = data[i];
        root = build(0,n,0);
    }

    std::vector<int> vec;

    void dfsvec(int p){
        if(p<0) return;
        vec.push_back(p);
        dfsvec(tree[p].l);
        dfsvec(tree[p].r);
    }

    // to make the nearing points into blocks
    void decorate(int p){
        if(p<0) return;
        ++nad;

        if(tree[p].R[0]-tree[p].L[0] < bx
            &&tree[p].R[1]-tree[p].L[1] < by
            &&tree[p].R[2]-tree[p].L[2] < bz){

            vec.clear();
            dfsvec(p);

            for(int it:vec){

                T dx = tree[it].pos[0] - tree[p].L[0];
                T dy = tree[it].pos[1] - tree[p].L[1];
                T dz = tree[it].pos[2] - tree[p].L[2];

                tree[p].q.push_back(dx+bx*dy+bx*by*dz);
            }

            tree[p].l = tree[p].r = -1;
        }
        else{
            decorate(tree[p].l);
            decorate(tree[p].r);
        }
    }

    void decorate(){
        nad=0;
        decorate(root);
        printf("ratio = %.6lf\n", 1.*n/nad);
    }

#define uchar SZ3::uchar

//    void save(uchar *bytes, SZ3::concepts::EncoderInterface<T>& encoder){
    void save(uchar *&bytes){
        switch(sizeof(size_t)){
            case 4:{
                int32ToBytes_bigEndian(bytes,tree[root].L[0]);
                bytes+=sizeof(size_t);
                int32ToBytes_bigEndian(bytes,tree[root].L[1]);
                bytes+=sizeof(size_t);
                int32ToBytes_bigEndian(bytes,tree[root].L[2]);
                bytes+=sizeof(size_t);
                int32ToBytes_bigEndian(bytes,tree[root].R[0]);
                bytes+=sizeof(size_t);
                int32ToBytes_bigEndian(bytes,tree[root].R[1]);
                bytes+=sizeof(size_t);
                int32ToBytes_bigEndian(bytes,tree[root].R[2]);
                bytes+=sizeof(size_t);
            }
            case 8:{
                int64ToBytes_bigEndian(bytes,tree[root].L[0]);
                bytes+=sizeof(size_t);
                int64ToBytes_bigEndian(bytes,tree[root].L[1]);
                bytes+=sizeof(size_t);
                int64ToBytes_bigEndian(bytes,tree[root].L[2]);
                bytes+=sizeof(size_t);
                int64ToBytes_bigEndian(bytes,tree[root].R[0]);
                bytes+=sizeof(size_t);
                int64ToBytes_bigEndian(bytes,tree[root].R[1]);
                bytes+=sizeof(size_t);
                int64ToBytes_bigEndian(bytes,tree[root].R[2]);
                bytes+=sizeof(size_t);
                break;
            }
            default:{
                exit(0);
            }
        }

        struct Node{
            int id;
            T L[3],R[3];
            Node(){
                id = -1;
                L[0] = L[1] = L[2] = 0;
                R[0] = R[1] = R[2] = 0;
            }
            Node(int id, KDNode<T>& node) :
                id(id) {
                L[0] = node.L[0];
                L[1] = node.L[1];
                L[2] = node.L[2];
                R[0] = node.R[0];
                R[1] = node.R[1];
                R[2] = node.R[2];
            }
            Node(int id, T *L_, T *R_) :
                id(id) {
                memcpy(L,L_,sizeof(L));
                memcpy(R,R_,sizeof(R));
            }
        };

        std::stack<Node> stk;

        stk.push(Node(root,tree[root]));

        int64_t *err = new int64_t[3*nad];
        int64_t *errh = new int64_t[nad];
        int64_t *blkcnt = new int64_t[nad];
        int64_t *token = new int64_t[n];

        int i=0,j=0;

        printf("begin to dfs\n");

        while(!stk.empty()){

            Node node = stk.top();
            int& u = node.id;
            stk.pop();

            int64_t predict_x = (node.R[0] - node.L[0]) / 2;
            int64_t predict_y = (node.R[1] - node.L[1]) / 2;
            int64_t predict_z = (node.R[2] - node.L[2]) / 2;

            err[i] = predict_x - tree[u].pos[0];
            err[nad+i] = predict_y - tree[u].pos[1];
            err[nad+nad+i] = predict_z - tree[u].pos[2];

            if(tree[u].q.empty()){
                blkcnt[i] = 1;
            }
            else{
                blkcnt[i] = tree[u].q.size();
                for(size_t it:tree[u].q){
                    if(!it) continue;
                    token[j] = it;
                    ++j;
                }

            }

            if(tree[u].r >= 0){
                stk.push(Node(tree[u].r,tree[u].pos.data(),node.R));
            }
            if(tree[u].l >= 0){
                stk.push(Node(tree[u].l,node.L,tree[u].pos.data()));
            }

            ++i;
        }

        T errL[3],errR[3];
        errL[0] = errR[0] = err[0];
        errL[1] = errR[1] = err[nad];
        errL[2] = errR[2] = err[nad+nad];

        for(int i=1;i<nad;i++){
            for(int j=0;j<3;j++){
                errL[j]=std::min(errL[j],err[j*nad+i]);
                errR[j]=std::max(errR[j],err[j*nad+i]);
            }
        }

        T errRange[3] = {errR[0] - errL[0] + 1, errR[1] - errL[1] + 1, errR[2] - errL[2] + 1};

        for(int i=0;i<3;i++){
            printf("errRange[%d] = %ld\n",i,errRange[i]);
        }

        for(int i=1;i<nad;i++){
            errh[i] = (err[i] - errL[0]) + (err[nad+i] - errL[1])*errRange[0] + (err[nad+nad+i] - errL[2])*errRange[0]*errRange[1];
        }

        delete[] err;

        SZ3::HuffmanEncoder<int64_t> encoder;

        uchar* head;

        printf("begin to save error\n");

//        head = bytes;
//        encoder.preprocess_encode(err,nad,0);
//        encoder.save(bytes);
//        encoder.encode(err,nad,bytes);
//        encoder.preprocess_encode(err+nad,nad,0);
//        encoder.save(bytes);
//        encoder.encode(err+nad,nad,bytes);
//        encoder.preprocess_encode(err+nad+nad,nad,0);
//        encoder.save(bytes);
//        encoder.encode(err+nad+nad,nad,bytes);
//        delete[] err;
//
//        printf("error size = %.6lf MB\n",1.*(bytes-head)/1024/1024);

//        head = bytes;
//        encoder.preprocess_encode(err,3*nad,0);
//        encoder.save(bytes);
//        encoder.encode(err,3*nad,bytes);
//        delete[] err;
//
//        printf("error size = %.6lf MB\n",1.*(bytes-head)/1024/1024);

        head = bytes;
        encoder.preprocess_encode(errh,nad,0);
        encoder.save(bytes);
        encoder.encode(errh,nad,bytes);
        delete[] errh;

        printf("error size = %.6lf MB\n",1.*(bytes-head)/1024/1024);

        printf("begin to save blkcnt\n");

        head = bytes;
        encoder.preprocess_encode(blkcnt,nad,0);
        encoder.save(bytes);
        encoder.encode(blkcnt,nad,bytes);
        delete[] blkcnt;

        printf("blkcnt size = %.6lf MB\n",1.*(bytes-head)/1024/1024);

        printf("begin to save tokens\n");

        head = bytes;
        encoder.preprocess_encode(token,j,0);
        encoder.save(bytes);
        encoder.encode(token,j,bytes);
        delete[] token;

        printf("token size = %.6lf MB\n",1.*(bytes-head)/1024/1024);

    }
};

}