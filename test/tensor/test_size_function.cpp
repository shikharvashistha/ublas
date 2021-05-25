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

BOOST_AUTO_TEST_SUITE(test_size_function, * boost::unit_test::description("Validate Size Function, Method, and Trait"))


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_function, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::size(extents_base<D> const&)")
    *boost::unit_test::description("Testing free function [size] for dynamic extents"))
{
    namespace ublas = boost::numeric::ublas;
    BOOST_TEST_CONTEXT("[Size Free Function] rank(0) dynamic extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n    ), 0ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(1) dynamic extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n1   ), 1ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n2   ), 1ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(2) dynamic extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n11  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n12  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n21  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n22  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n32  ), 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(3) dynamic extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n111 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n112 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n121 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n123 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n211 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n213 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n321 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n432 ), 3ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(4) dynamic extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n1111), 4ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n4231), 4ul);
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_function,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::size(extents_base<D> const&)")
    *boost::unit_test::description("Testing free function [size] for static rank extents"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Free Function] rank(0) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n    ), 0ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(1) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n1   ), 1ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n2   ), 1ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(2) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n11  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n12  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n21  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n22  ), 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n32  ), 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(3) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n111 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n112 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n121 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n123 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n211 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n213 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n321 ), 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n432 ), 3ul);
    }

    BOOST_TEST_CONTEXT("[Size Free Function] rank(4) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size(n1111), 4ul);
        BOOST_REQUIRE_EQUAL(ublas::size(n4231), 4ul);
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_trait,
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::size_v<extents_core<...>>")
    *boost::unit_test::description("Testing trait [size_v] for static extents"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Trait] rank(0) static rank extents"){
        BOOST_REQUIRE_EQUAL(ublas::size_v<n_type    >, 0ul);
    }

    BOOST_TEST_CONTEXT("[Size Trait] rank(1) static extents"){
        BOOST_REQUIRE_EQUAL(ublas::size_v<n1_type   >, 1ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n2_type   >, 1ul);
    }

    BOOST_TEST_CONTEXT("[Size Trait] rank(2) static extents"){
        BOOST_REQUIRE_EQUAL(ublas::size_v<n11_type  >, 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n12_type  >, 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n21_type  >, 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n22_type  >, 2ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n32_type  >, 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Trait] rank(3) static extents"){
        BOOST_REQUIRE_EQUAL(ublas::size_v<n111_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n112_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n121_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n123_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n211_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n213_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n321_type >, 3ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n432_type >, 3ul);
    }

    BOOST_TEST_CONTEXT("[Size Trait] rank(4) static extents"){
        BOOST_REQUIRE_EQUAL(ublas::size_v<n1111_type>, 4ul);
        BOOST_REQUIRE_EQUAL(ublas::size_v<n4231_type>, 4ul);
    }
}

BOOST_FIXTURE_TEST_CASE(test_tensor_dynamic,
    boost::numeric::ublas::fixture_tensor_dynamic<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::size")
    *boost::unit_test::description("Testing the dynamic tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Method] rank(1) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static_rank,
    boost::numeric::ublas::fixture_tensor_static_rank<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::size")
    *boost::unit_test::description("Testing the static rank tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Method] rank(1) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static,
    boost::numeric::ublas::fixture_tensor_static<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::size")
    *boost::unit_test::description("Testing the static tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Method] rank(1) static tensor"){
        BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) static tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) static tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) static tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}


BOOST_AUTO_TEST_SUITE_END()
