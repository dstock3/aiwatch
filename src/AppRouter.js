import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './components/pages/Home';
import About from './components/pages/About';
import Videos from './components/pages/Videos';
import Blog from './components/pages/Blog';
import Resources from './components/pages/Resources';
import Contact from './components/pages/Contact';

const AppRouter = () => (
  <Router>
    <Switch>
      <Route exact path="/" component={Home} />
      <Route path="/about" component={About} />
      <Route path="/videos" component={Videos} />
      <Route path="/blog" component={Blog} />
      <Route path="/resources" component={Resources} />
      <Route path="/contact" component={Contact} />
    </Switch>
  </Router>
);

export default AppRouter;
