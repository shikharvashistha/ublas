//
// 	Copyright (c) 2021  Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include "fixture_utility.hpp"

BOOST_AUTO_TEST_SUITE(test_access_operator, 
    *boost::unit_test::description("Validate Access Operator")
    *boost::unit_test::depends_on("test_size_function")
)


constexpr auto check_at(auto const& e, std::initializer_list<std::size_t> l){
    auto i = 0ul;
    for(auto const& el : l){
        BOOST_REQUIRE_EQUAL(e[i++], el);
    }
};

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_function, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<>::operator[]")
    *boost::unit_test::description("Testing the dynamic extents' read [operator[]]"))
{
    namespace ublas = boost::numeric::ublas;


    BOOST_TEST_CONTEXT("[At Method] rank(1) dynamic extents"){
        check_at(n1, {1});
        check_at(n2, {2});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(2) dynamic extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) dynamic extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) dynamic extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }

}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_function,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<N>::operator[]")
    *boost::unit_test::description("Testing the static rank extents' read [operator[]]"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[At Method] rank(1) static rank extents"){
        check_at(n1, {1});
        check_at(n2, {2});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(2) static rank extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) static rank extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) static rank extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_function,
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<...>::operator[]")
    *boost::unit_test::description("Testing the static extents' read [operator[]]"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[At Method] rank(2) static extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) static extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) static extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::operator()/operator[]")
    *boost::unit_test::description("Testing the dynamic tensor's read/write [operator()/operator[]] for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") dynamic tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
            
            auto w = v + value_type(1);
            BOOST_TEST_CHECKPOINT("[operator[]] rank("<< rank <<") dynamic tensor operator[](" << i << ") = " << w);
            t[i] = w;
            BOOST_REQUIRE_EQUAL(t[i], w);
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::operator()/operator[]")
    *boost::unit_test::description("Testing the dynamic tensor's read/write [operator()] for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") dynamic tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") dynamic tensor operator()(" << i <<", "<< j << ") = " << v);
                t(i,j) = v;
                BOOST_REQUIRE_EQUAL(t(i,j), v);
                v += value_type(1);
            }
        }
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") dynamic tensor operator()(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") dynamic tensor operator()(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4](auto /*id*/, auto const& t){
        if(t.rank() == 1) check1(t);
        else if(t.rank() == 2) check2(t);
        else if(t.rank() == 3) check3(t);
        else if(t.rank() == 4) check4(t);
        else assert(false && "Not Implemented");

    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::operator()/operator[]")
    *boost::unit_test::description("Testing the static rank tensor's read/write [operator()/operator[]] for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static rank tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
                        
            auto w = v + value_type(1);
            BOOST_TEST_CHECKPOINT("[operator[]] rank("<< rank <<") static rank tensor operator[](" << i << ") = " << w);
            t[i] = w;
            BOOST_REQUIRE_EQUAL(t[i], w);
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::operator()/operator[]")
    *boost::unit_test::description("Testing the static rank tensor's read/write [operator()] for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[operator()] static rank("<< rank <<") rank tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[operator()] static rank("<< rank <<") rank tensor operator()(" << i <<", "<< j << ") = " << v);
                t(i,j) = v;
                BOOST_REQUIRE_EQUAL(t(i,j), v);
                v += value_type(1);
            }
        }
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[operator()] static rank("<< rank <<") rank tensor operator()(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[operator()] static rank("<< rank <<") rank tensor operator()(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4]<typename TensorType>(auto /*id*/, TensorType const& t){
        constexpr auto e = typename TensorType::extents_type{};
        if constexpr(ublas::size(e) == 1) check1(t);
        else if constexpr(ublas::size(e) == 2) check2(t);
        else if constexpr(ublas::size(e) == 3) check3(t);
        else if constexpr(ublas::size(e) == 4) check4(t);
        else assert(false && "Not Implemented");

    });
}


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::operator()/operator[]")
    *boost::unit_test::description("Testing the static tensor's read/write [operator()/operator[]] for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
                        
            auto w = v + value_type(1);
            BOOST_TEST_CHECKPOINT("[operator[]] rank("<< rank <<") static tensor operator[](" << i << ") = " << w);
            t[i] = w;
            BOOST_REQUIRE_EQUAL(t[i], w);
        }
    };
    
    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::operator()/operator[]")
    *boost::unit_test::description("Testing the static tensor's read/write [operator()] for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static tensor operator()(" << i << ") = " << v);
            t(i) = v;
            BOOST_REQUIRE_EQUAL(t(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static tensor operator()(" << i <<", "<< j << ") = " << v);
                t(i,j) = v;
                BOOST_REQUIRE_EQUAL(t(i,j), v);
                v += value_type(1);
            }
        }
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static tensor operator()(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[operator()] rank("<< rank <<") static tensor operator()(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4]<typename TensorType>(auto /*id*/, TensorType const& t){
        using extents_type = typename TensorType::extents_type;
        if constexpr(ublas::size_v<extents_type> == 1) check1(t);
        else if constexpr(ublas::size_v<extents_type> == 2) check2(t);
        else if constexpr(ublas::size_v<extents_type> == 3) check3(t);
        else if constexpr(ublas::size_v<extents_type> == 4) check4(t);
        else assert(false && "Not Implemented");
    });
}


BOOST_AUTO_TEST_SUITE_END()
